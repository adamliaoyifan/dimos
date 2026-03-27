# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Stuck / wall escape skill for DimOS.

Monitors the robot's movement and detects when it is stuck against a wall or
obstacle. When stuck, uses the VLM to confirm the situation and then executes
an escape maneuver (back up + rotate toward open space).

Works with the existing A* planner — injects escape goals via NavigationInterface
or publishes relative_move commands via UnitreeSkillContainer RPCs.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


class EscapeConfig(ModuleConfig):
    """Configuration for stuck detection and escape."""

    stuck_check_interval: float = 2.0
    """Seconds between stuck checks."""

    stuck_distance_threshold: float = 0.05
    """If robot moves less than this (meters) in stuck_time_window, it's stuck."""

    stuck_time_window: float = 6.0
    """Time window (seconds) to evaluate if robot is stuck."""

    escape_backup_distance: float = 0.3
    """How far to back up (meters) when stuck."""

    escape_rotate_degrees: float = 90.0
    """How much to rotate (degrees) when stuck to find open space."""

    max_escape_attempts: int = 4
    """Max escape attempts before giving up and reporting to agent."""

    vlm_backend: str = "qwen3_local"
    """VLM backend for wall detection."""

    vlm_base_url: str = "http://192.168.2.109:8000"
    """VLM server URL."""

    enable_vlm_check: bool = True
    """Whether to use VLM to confirm wall/obstacle before escaping."""

    vlm_prompt_prefix: str = ""
    """Prefix prepended to every VLM prompt (e.g. simulation context)."""


class EscapeSkillContainer(Module[EscapeConfig]):
    """Detects when the robot is stuck against a wall and escapes.

    Monitors odometry to detect lack of progress. When stuck:
    1. Optionally uses VLM to confirm wall/obstacle ahead
    2. Backs up and rotates to find open space
    3. Reports the situation to the agent

    Runs a background monitoring thread that automatically triggers
    escape when the robot appears stuck during active navigation.
    """

    default_config = EscapeConfig

    rpc_calls: list[str] = [
        "NavigationInterface.set_goal",
        "NavigationInterface.get_state",
        "NavigationInterface.cancel_goal",
    ]

    color_image: In[Image]
    odom: In[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_image: Image | None = None
        self._latest_odom: PoseStamped | None = None
        self._started = False
        self._escaping = False
        self._escape_count = 0

        # Position history for stuck detection
        self._position_history: list[tuple[float, float, float]] = []  # (x, y, timestamp)
        self._lock = threading.Lock()

        # Monitor thread
        self._monitor_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # VLM for wall confirmation
        self._vl_model = None

    def _create_vlm(self) -> Any:
        """Create VLM for wall detection."""
        if self.config.vlm_backend == "qwen3_local":
            from dimos.models.vl.qwen3_local import Qwen3LocalVlModel
            return Qwen3LocalVlModel(
                base_url=self.config.vlm_base_url,
                prompt_prefix=self.config.vlm_prompt_prefix,
            )
        elif self.config.vlm_backend == "qwen_local":
            from dimos.models.vl.qwen_local import QwenLocalVlModel
            return QwenLocalVlModel(base_url=self.config.vlm_base_url)
        else:
            from dimos.models.vl.qwen import QwenVlModel
            return QwenVlModel()

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.color_image.subscribe(self._on_image)))
        self._disposables.add(Disposable(self.odom.subscribe(self._on_odom)))

        if self.config.enable_vlm_check:
            try:
                self._vl_model = self._create_vlm()
            except Exception as e:
                logger.warning(f"[Escape] Could not create VLM for wall detection: {e}")

        self._started = True
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="escape-monitor"
        )
        self._monitor_thread.start()
        logger.info("[Escape] Stuck detection + escape skill started")

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=3.0)
        self._started = False
        super().stop()

    def _on_image(self, image: Image) -> None:
        self._latest_image = image

    def _on_odom(self, odom: PoseStamped) -> None:
        self._latest_odom = odom
        with self._lock:
            now = time.monotonic()
            self._position_history.append((odom.position.x, odom.position.y, now))
            # Prune old entries
            cutoff = now - self.config.stuck_time_window * 2
            self._position_history = [
                p for p in self._position_history if p[2] > cutoff
            ]

    def _is_stuck(self) -> bool:
        """Check if robot hasn't moved enough in the time window."""
        with self._lock:
            if len(self._position_history) < 3:
                return False

            now = time.monotonic()
            cutoff = now - self.config.stuck_time_window
            recent = [p for p in self._position_history if p[2] > cutoff]

            if len(recent) < 2:
                return False

            # Check total displacement (not path length)
            x0, y0, _ = recent[0]
            x1, y1, _ = recent[-1]
            displacement = math.hypot(x1 - x0, y1 - y0)

            return displacement < self.config.stuck_distance_threshold

    def _is_wall_ahead(self) -> bool:
        """Use VLM to check if there's a wall/obstacle directly ahead."""
        if self._vl_model is None or self._latest_image is None:
            return True  # Assume wall if we can't check

        try:
            prompt = (
                "Look at this image from a robot's camera. "
                "Is the robot facing a wall, obstacle, or dead end that blocks forward movement? "
                "Answer with ONLY 'yes' or 'no'."
            )
            response = self._vl_model.query(self._latest_image, prompt)
            return "yes" in response.lower().split()
        except Exception as e:
            logger.warning(f"[Escape] VLM wall check failed: {e}")
            return True  # Assume wall on failure

    def _find_open_direction(self) -> str:
        """Use VLM to determine which direction has the most open space."""
        if self._vl_model is None or self._latest_image is None:
            return "left"  # Default

        try:
            prompt = (
                "The robot is stuck facing an obstacle. "
                "Looking at the current camera view, which direction has more open space "
                "for the robot to escape: 'left' or 'right'? "
                "Answer with ONLY 'left' or 'right'."
            )
            response = self._vl_model.query(self._latest_image, prompt)
            if "right" in response.lower():
                return "right"
            return "left"
        except Exception:
            return "left"

    def _execute_escape(self) -> str:
        """Execute escape maneuver: back up + rotate toward open space."""
        self._escaping = True
        self._escape_count += 1

        try:
            # Step 1: Cancel current navigation goal
            try:
                cancel_rpc = self.get_rpc_calls("NavigationInterface.cancel_goal")
                cancel_rpc()
            except Exception:
                pass

            logger.info("[Escape] Starting escape maneuver (attempt %d)", self._escape_count)

            # Step 2: Check if wall ahead (VLM)
            wall_ahead = self._is_wall_ahead()
            if not wall_ahead:
                logger.info("[Escape] VLM says no wall ahead — false alarm, resuming")
                self._escaping = False
                return "Not stuck against wall, resuming navigation."

            # Step 3: Determine escape direction
            open_direction = self._find_open_direction()
            logger.info(f"[Escape] Wall confirmed. Open direction: {open_direction}")

            # Step 4: Back up
            backup_dist = self.config.escape_backup_distance
            logger.info(f"[Escape] Backing up {backup_dist}m")

            try:
                set_goal_rpc = self.get_rpc_calls("NavigationInterface.set_goal")
                if self._latest_odom:
                    # Compute a goal point behind the robot
                    q = self._latest_odom.orientation
                    siny = 2.0 * (q.w * q.z + q.x * q.y)
                    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
                    yaw = math.atan2(siny, cosy)

                    back_x = self._latest_odom.position.x - backup_dist * math.cos(yaw)
                    back_y = self._latest_odom.position.y - backup_dist * math.sin(yaw)

                    from dimos.msgs.geometry_msgs.Quaternion import Quaternion
                    from dimos.msgs.geometry_msgs.Vector3 import make_vector3

                    goal = PoseStamped(
                        position=make_vector3(back_x, back_y, 0),
                        orientation=Quaternion.from_euler(make_vector3(0, 0, yaw)),
                        frame_id="map",
                    )
                    set_goal_rpc(goal)

                    # Wait for backup to complete
                    time.sleep(2.0)

                    # Step 5: Rotate toward open space
                    rotate_rad = math.radians(self.config.escape_rotate_degrees)
                    if open_direction == "right":
                        rotate_rad = -rotate_rad

                    new_yaw = yaw + rotate_rad
                    rotate_goal = PoseStamped(
                        position=make_vector3(back_x, back_y, 0),
                        orientation=Quaternion.from_euler(make_vector3(0, 0, new_yaw)),
                        frame_id="map",
                    )
                    set_goal_rpc(rotate_goal)

                    time.sleep(2.0)

                    logger.info("[Escape] Escape maneuver complete")
                    return (
                        f"Escaped from wall. Backed up {backup_dist}m and rotated "
                        f"{self.config.escape_rotate_degrees}° {open_direction}."
                    )
            except Exception as e:
                logger.error(f"[Escape] Escape maneuver failed: {e}")
                return f"Escape attempt failed: {e}"

            return "Could not determine position for escape."
        finally:
            self._escaping = False

    def _monitor_loop(self) -> None:
        """Background thread that checks if robot is stuck."""
        while not self._stop_event.is_set():
            self._stop_event.wait(timeout=self.config.stuck_check_interval)
            if self._stop_event.is_set():
                break

            # Only check when we're navigating (not idle, not already escaping)
            if self._escaping:
                continue

            try:
                get_state_rpc = self.get_rpc_calls("NavigationInterface.get_state")
                from dimos.navigation.base import NavigationState
                state = get_state_rpc()
                if state == NavigationState.IDLE:
                    self._escape_count = 0  # Reset counter when idle
                    continue
            except Exception:
                continue

            if self._is_stuck():
                if self._escape_count >= self.config.max_escape_attempts:
                    logger.warning(
                        "[Escape] Max escape attempts (%d) reached. Cancelling navigation.",
                        self.config.max_escape_attempts,
                    )
                    try:
                        cancel_rpc = self.get_rpc_calls("NavigationInterface.cancel_goal")
                        cancel_rpc()
                    except Exception:
                        pass
                    self._escape_count = 0
                    continue

                logger.info("[Escape] Robot appears stuck, initiating escape")
                self._execute_escape()

    # ------------------------------------------------------------------
    # Skills exposed to agent
    # ------------------------------------------------------------------

    @skill
    def check_if_stuck(self) -> str:
        """Check if the robot is currently stuck against a wall or obstacle.

        Analyzes recent movement history and optionally uses the camera to
        confirm if there's a wall ahead. Use this if the robot seems to not
        be making progress toward its goal.

        Returns:
            str: Status description including whether robot is stuck and what it sees.
        """
        if not self._started:
            return "Error: Escape skill not started."

        is_stuck = self._is_stuck()
        wall_ahead = self._is_wall_ahead() if is_stuck else False

        if is_stuck and wall_ahead:
            return (
                "The robot IS stuck — it hasn't moved significantly and there's "
                "a wall/obstacle ahead. Call 'escape_from_wall' to recover."
            )
        elif is_stuck:
            return (
                "The robot hasn't moved much recently but no wall is detected ahead. "
                "It may be stuck in a planner loop. Try 'escape_from_wall' or re-issue the goal."
            )
        return "The robot is moving normally, not stuck."

    @skill
    def escape_from_wall(self) -> str:
        """Execute an escape maneuver when the robot is stuck against a wall.

        The robot will:
        1. Cancel current navigation goal
        2. Use the camera to confirm a wall/obstacle ahead
        3. Back up away from the wall
        4. Rotate toward the direction with more open space
        5. Resume navigation

        Call this when the robot is stuck against a wall or not making progress.

        Returns:
            str: Description of escape outcome.
        """
        if not self._started:
            return "Error: Escape skill not started."

        if self._escaping:
            return "An escape maneuver is already in progress."

        return self._execute_escape()


__all__ = ["EscapeSkillContainer"]

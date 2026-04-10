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

"""Vision-and-Language Navigation (VLN) skill for compound goals.

Decomposes compound goals like "find the glasses box in the CTO office room"
into a hierarchical search: first navigate to the room, then actively search
for the target object by combining frontier exploration with continuous VLM
checking.
"""

from __future__ import annotations

import json
import math
import re
import threading
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In
from dimos.models.qwen.bbox import BBox
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3, make_vector3
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.base import NavigationState
from dimos.navigation.visual.query import get_object_bbox_from_image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()
T = TypeVar("T")


class VLNConfig(ModuleConfig):
    """Configuration for the VLN skill."""

    vlm_check_interval: float = 1.0
    """Seconds between VLM checks during active search."""

    search_timeout: float = 300.0
    """Maximum seconds to search before giving up."""

    approach_timeout: float = 30.0
    """Maximum seconds to approach a detected object."""

    similarity_threshold: float = 0.23
    """Minimum CLIP similarity for semantic map queries."""

    exploration_mode: str = "astar"
    """Exploration backend: "astar" (WavefrontFrontier + A*) or "navdp" (diffusion-policy nogoal).
    Use "navdp" to test the NavDP server in isolation."""

    vlm_backend: str = "qwen3_local"
    """VLM backend: 'qwen3_local' (custom Qwen3 server), 'qwen_local' (OpenAI-compat),
    'qwen' (Alibaba API), or 'moondream' (local HF)."""

    vlm_base_url: str = "http://192.168.2.109:8000"
    """Base URL for the VLM server."""

    vlm_model_name: str = "Qwen3-VL-8B-Instruct"
    """Model name (used by qwen_local backend only)."""

    vlm_prompt_prefix: str = ""
    """Prefix prepended to every VLM prompt (e.g. simulation context)."""

    confirm_checks: int = 3
    """Number of VLM checks per heading during detection confirmation."""

    confirm_threshold: int = 2
    """Minimum detections out of confirm_checks required to confirm the object."""

    confirm_check_delay: float = 1.5
    """Seconds to wait between VLM checks during confirmation (ensures fresh camera frame)."""

    navdp_imagegoal_debug_dir: str = ""
    """If non-empty, save the BGR image passed to NavDP set_reference_image
    (bbox crop when available) as PNG under this directory."""

    navdp_imagegoal_width: int = 640
    """Width to resize the NavDP imagegoal before sending to set_reference_image.
    Should match the NavDP trajectory camera resolution (640 for Go2, adjust for RealSense)."""

    navdp_imagegoal_height: int = 480
    """Height to resize the NavDP imagegoal before sending to set_reference_image.
    Should match the NavDP trajectory camera resolution (480 for Go2, adjust for RealSense)."""

    approach_stop_distance: float = 0.6
    """Distance from the estimated object centre to stop at during A* approach (metres)."""

    approach_min_depth_confidence: float = 0.3
    """Minimum valid-pixel fraction required to trust a depth-based 3D estimate."""

    approach_ema_alpha: float = 0.3
    """EMA learning rate for updating the 3D target position during approach."""

    cam_intrinsic: list[list[float]] | None = None
    """3×3 camera intrinsic matrix.  Set from blueprint config at start()."""

    cam_x: float = 0.13
    """Camera x offset in base_link frame (forward, metres)."""

    cam_y: float = 0.00
    """Camera y offset in base_link frame (left, metres)."""

    cam_z: float = 0.30
    """Camera z offset in base_link frame (up, metres)."""

    cam_pitch: float = 0.157
    """Camera downward tilt angle in radians (~9°)."""

    depth_scale: float = 1.0
    """Multiplier to convert raw depth pixel values to metres (1.0 for simulation,
    0.001 for sensors that report depth in mm)."""

    search_stall_timeout: float = 120.0
    """Give up when the robot stops covering new ground for this many seconds."""

    search_progress_distance: float = 1.0
    """Metres of new trail distance needed within each stall window to count as progress."""

    search_observed_fraction: float = 0.85
    """If the costmap observed fraction exceeds this value the boundary is considered fully
    explored and the search terminates early."""


class VLNSkillContainer(Module[VLNConfig]):
    """Vision-and-Language Navigation skill for compound goals.

    Adds three capabilities missing from the base NavigationSkillContainer:

    1. **Goal decomposition** — breaks "glasses box in CTO office" into
       [room: "CTO office", object: "glasses box"] using the VLM.
    2. **Room identification** — uses VLM scene captioning during exploration
       to recognize when the robot has entered the target room.
    3. **Active object search** — continuously checks camera frames for the
       target object while the robot explores, instead of only checking once.

    Requires the following modules in the same blueprint:
        - NavigationInterface (A* planner)
        - WavefrontFrontierExplorer
        - SpatialMemory
        - ObjectTracking
    """

    default_config = VLNConfig

    rpc_calls: list[str] = [
        "ReplanningAStarPlanner.set_goal",
        "ReplanningAStarPlanner.get_state",
        "ReplanningAStarPlanner.is_goal_reached",
        "ReplanningAStarPlanner.cancel_goal",
        # "WavefrontFrontierExplorer.explore",
        # "WavefrontFrontierExplorer.stop_exploration",
        # "WavefrontFrontierExplorer.is_exploration_active",
        "SpatialMemory.query_by_text",
        "SpatialMemory.tag_location",
        "SpatialMemory.query_tagged_location",
        # NavDP (optional — used for imagegoal approach when available)
        "NavDPNavigator.set_language_goal",
        "NavDPNavigator.set_reference_image",
        "NavDPNavigator.get_navdp_state",
        "NavDPNavigator.pause_motion",
        "NavDPNavigator.resume_motion",
        "NavDPNavigator.cancel_goal",
        "NavDPNavigator.get_exploration_stats",
        "NavDPNavigator.clear_exploration_trail",
        # In-place rotation during detection confirmation (uses A* planner, collision-safe)
        "UnitreeSkillContainer.relative_move",
        # NavDP memory — object instance recording (optional)
        "NavDPMemory.record_object_instance",
        "NavDPMemory.query_object_3d",
    ]

    color_image: In[Image]
    depth_image: In[Image]
    odom: In[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vl_model = self._create_vlm()

        self._latest_image: Image | None = None
        self._latest_depth: Image | None = None
        self._latest_odom: PoseStamped | None = None
        self._started = False

        # ObjectLocalizer — initialised in start() once config is available
        from dimos.navigation.visual.object_localizer import ObjectLocalizer
        self._localizer: ObjectLocalizer | None = None

        # Active search state
        self._search_active = False
        self._search_stop = threading.Event()
        self._search_thread: threading.Thread | None = None
        self._search_result: str | None = None

    def _create_vlm(self):  # type: ignore[no-untyped-def]
        """Instantiate the VLM backend based on config."""
        backend = self.config.vlm_backend
        if backend == "qwen3_local":
            from dimos.models.vl.qwen3_local import Qwen3LocalVlModel

            return Qwen3LocalVlModel(
                base_url=self.config.vlm_base_url,
                prompt_prefix=self.config.vlm_prompt_prefix,
            )
        elif backend == "qwen_local":
            from dimos.models.vl.qwen_local import QwenLocalVlModel

            return QwenLocalVlModel(
                base_url=self.config.vlm_base_url,
                model_name=self.config.vlm_model_name,
            )
        elif backend == "moondream":
            from dimos.models.vl.moondream import MoondreamVlModel

            return MoondreamVlModel()
        else:
            from dimos.models.vl.qwen import QwenVlModel

            return QwenVlModel()

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.color_image.subscribe(self._on_image)))
        self._disposables.add(Disposable(self.depth_image.subscribe(self._on_depth)))
        self._disposables.add(Disposable(self.odom.subscribe(self._on_odom)))

        # Initialise ObjectLocalizer with camera intrinsics from config
        from dimos.navigation.visual.object_localizer import ObjectLocalizer
        import numpy as np

        if self.config.cam_intrinsic is not None:
            K = np.array(self.config.cam_intrinsic, dtype=np.float64)
        else:
            # Default Go2 intrinsics
            K = np.array([
                [460.0, 0.0, 320.0],
                [0.0, 460.0, 240.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
        self._localizer = ObjectLocalizer(
            intrinsic=K,
            cam_x=self.config.cam_x,
            cam_y=self.config.cam_y,
            cam_z=self.config.cam_z,
            cam_pitch=self.config.cam_pitch,
            depth_scale=self.config.depth_scale,
        )

        self._started = True
        logger.info(
            "[VLN] started — exploration_mode=%s, vlm=%s, localizer=ready",
            self.config.exploration_mode, self.config.vlm_backend,
        )

    @rpc
    def stop(self) -> None:
        self._cancel_search()
        super().stop()

    def _on_image(self, image: Image) -> None:
        self._latest_image = image

    def _on_depth(self, image: Image) -> None:
        self._latest_depth = image

    def _on_odom(self, odom: PoseStamped) -> None:
        self._latest_odom = odom

    # ------------------------------------------------------------------
    # Goal decomposition
    # ------------------------------------------------------------------

    def _decompose_goal(self, goal: str) -> dict[str, str | None]:
        """Use the VLM to decompose a compound goal into room + object.

        Returns dict with keys 'room' and 'object'.  Either may be None
        if the goal is simple (e.g. just an object or just a room).
        """
        if self._latest_image is None:
            # No image available yet — fall back to text-only decomposition
            return self._decompose_goal_text_only(goal)

        prompt = (
            "You are a goal-decomposition assistant for a mobile robot.\n"
            f"The user asked the robot to: \"{goal}\"\n\n"
            "Break this into TWO parts:\n"
            "1. 'room' — the room or area to go to (or null if not specified)\n"
            "2. 'object' — the specific object to find (or null if not specified)\n\n"
            "Return ONLY a JSON object, e.g.:\n"
            '{"room": "CTO office", "object": "glasses box"}\n'
            "If the goal is just a room, set object to null.\n"
            "If the goal is just an object with no room, set room to null."
        )

        from dimos.utils.generic import extract_json_from_llm_response

        response = self._vl_model.query(self._latest_image, prompt)
        result = extract_json_from_llm_response(response)
        if result and isinstance(result, dict):
            return {
                "room": result.get("room"),
                "object": result.get("object"),
            }

        return self._decompose_goal_text_only(goal)

    def _decompose_goal_text_only(self, goal: str) -> dict[str, str | None]:
        """Simple heuristic decomposition when VLM is unavailable."""
        goal_lower = goal.lower()
        # Common patterns: "X in Y", "X in the Y"
        for sep in [" in the ", " in "]:
            if sep in goal_lower:
                idx = goal_lower.index(sep)
                obj_part = goal[:idx].strip()
                room_part = goal[idx + len(sep) :].strip()
                if obj_part and room_part:
                    return {"room": room_part, "object": obj_part}
        # No decomposition possible — treat whole goal as object
        return {"room": None, "object": goal}

    # ------------------------------------------------------------------
    # Room identification via VLM
    # ------------------------------------------------------------------

    def _stop_motion_for_vlm_check(self) -> None:
        """Best-effort hard stop before a blocking VLM request."""
        # Stop A* motion if active.
        try:
            cancel_rpc = self.get_rpc_calls("ReplanningAStarPlanner.cancel_goal")
            cancel_rpc()
        except Exception:
            pass

        # Pause NavDP motion without resetting stuck/escape state.
        try:
            navdp_pause = self.get_rpc_calls("NavDPNavigator.pause_motion")
            navdp_pause()
        except Exception:
            pass

        # NavDP exploration must stay active across checks so its stuck/escape
        # history can accumulate; only pause motion. A* exploration still needs
        # to be stopped explicitly because it has no pause primitive.
        if self.config.exploration_mode != "navdp":
            try:
                self._stop_exploration()
            except Exception:
                pass
        else:
            # #region agent log
            try:
                import json as _json
                with open("/home/adamliao/work/dimos/.cursor/debug.log", "a") as _dbgf:
                    _dbgf.write(_json.dumps({
                        "timestamp": int(time.time() * 1000),
                        "runId": "post-fix",
                        "hypothesisId": "A-reset-loop",
                        "location": "vln_skill.py:pause_for_vlm_navdp",
                        "message": "paused NavDP motion without canceling exploration",
                    }) + "\n")
            except Exception:
                pass
            # #endregion

    def _resume_motion_after_vlm_check(self) -> None:
        """Best-effort resume for motion previously paused during VLM checks."""
        try:
            navdp_resume = self.get_rpc_calls("NavDPNavigator.resume_motion")
            navdp_resume()
        except Exception:
            pass

    def _update_navdp_object_direction(self, bbox: "BBox") -> None:
        """Classify bbox as left/centre/right and push to NavDP trajectory selector."""
        if self._latest_image is None:
            return
        img_w = self._latest_image.data.shape[1]
        cx = (bbox[0] + bbox[2]) / 2.0
        if bbox[2] > img_w:  # 0-1000 normalised scale
            cx = cx / 1000.0 * img_w
        third = img_w / 3.0
        if cx < third:
            direction = "left"
        elif cx > 2 * third:
            direction = "right"
        else:
            direction = "centre"
        logger.info("[VLN] Object direction: %s (cx=%.0f/%d)", direction, cx, img_w)
        try:
            rpc = self.get_rpc_calls("NavDPNavigator.set_object_direction")
            rpc(direction)
        except Exception:
            pass

    def _clear_navdp_object_direction(self) -> None:
        """Clear the direction hint on the NavDP trajectory selector."""
        try:
            rpc = self.get_rpc_calls("NavDPNavigator.clear_object_direction")
            rpc()
        except Exception:
            pass

    def _resume_exploration_after_vlm_check(self) -> None:
        """Resume exploration after a blocking VLM check."""
        self._resume_motion_after_vlm_check()
        if self.config.exploration_mode != "navdp":
            self._start_exploration()
        else:
            # #region agent log
            try:
                import json as _json
                with open("/home/adamliao/work/dimos/.cursor/debug.log", "a") as _dbgf:
                    _dbgf.write(_json.dumps({
                        "timestamp": int(time.time() * 1000),
                        "runId": "post-fix",
                        "hypothesisId": "A-reset-loop",
                        "location": "vln_skill.py:resume_after_vlm_navdp",
                        "message": "resumed NavDP motion without restarting exploration",
                    }) + "\n")
            except Exception:
                pass
            # #endregion

    def _run_vlm_check_while_paused(self, check_fn: Callable[[], T]) -> T:
        """Pause robot motion, run blocking VLM check, then return its result."""
        self._stop_motion_for_vlm_check()
        # Give the robot and camera a brief settle window.
        time.sleep(0.15)
        return check_fn()

    def _check_room_match(self, room_description: str, *, resume_search: bool = False) -> bool:
        """Ask the VLM whether the current camera view matches the room."""

        def _query() -> bool:
            if self._latest_image is None:
                return False

            prompt = (
                f'Look at this image. Is this a "{room_description}"?\n'
                "Answer with ONLY 'yes' or 'no'."
            )
            response = self._vl_model.query(self._latest_image, prompt)
            return "yes" in response.lower().split()

        matched = self._run_vlm_check_while_paused(_query)
        if resume_search and not matched and not self._search_stop.is_set():
            self._resume_exploration_after_vlm_check()
        return matched

    # ------------------------------------------------------------------
    # Object detection via VLM
    # ------------------------------------------------------------------

    def _check_object_in_view(
        self, object_description: str, *, resume_search: bool = False
    ) -> tuple["BBox | None", "PoseStamped | None"]:
        """Check if the target object is visible in the current frame.

        Returns (bbox, capture_odom) where capture_odom is the robot pose at
        the moment the image was captured (before the blocking VLM call).
        Callers can compare capture_odom to the current odom to detect overrun.
        """
        def _query() -> tuple["BBox | None", "PoseStamped | None"]:
            if self._latest_image is None:
                return None, None
            # Snapshot odom at capture time — BEFORE the blocking VLM call
            capture_odom = self._latest_odom
            t0 = time.time()
            try:
                result = get_object_bbox_from_image(
                    self._vl_model, self._latest_image, object_description
                )
            except Exception as exc:
                elapsed_ms = (time.time() - t0) * 1000
                logger.info(
                    "[VLN] VLM check '%s': EXCEPTION after %.0fms — %s",
                    object_description, elapsed_ms, exc,
                )
                return None, capture_odom
            elapsed_ms = (time.time() - t0) * 1000
            # Log current position (at result time) for diagnostics
            _pos = (
                (round(self._latest_odom.position.x, 2), round(self._latest_odom.position.y, 2))
                if self._latest_odom else None
            )
            _cap_pos = (
                (round(capture_odom.position.x, 2), round(capture_odom.position.y, 2))
                if capture_odom else None
            )
            logger.info(
                "[VLN] VLM check '%s': %s (%.0fms) pos=%s cap_pos=%s",
                object_description,
                f"FOUND bbox={result}" if result else "not found",
                elapsed_ms,
                _pos,
                _cap_pos,
            )
            return result, capture_odom

        result = self._run_vlm_check_while_paused(_query)
        if resume_search and result[0] is None and not self._search_stop.is_set():
            self._resume_exploration_after_vlm_check()
        return result

    def _compute_overrun(self, capture_odom: "PoseStamped | None") -> float:
        """Euclidean distance between the current odom and where the image was captured."""
        if capture_odom is None or self._latest_odom is None:
            return 0.0
        dx = self._latest_odom.position.x - capture_odom.position.x
        dy = self._latest_odom.position.y - capture_odom.position.y
        return math.sqrt(dx * dx + dy * dy)

    def _navigate_to_capture_pose(self, capture_odom: "PoseStamped | None") -> None:
        """Drive the robot back to *capture_odom* using ``relative_move``.

        Computes the displacement in the robot's current local frame and
        issues a single relative_move command.  Waits for the move to
        complete before returning.
        """
        if capture_odom is None or self._latest_odom is None:
            return

        cur = self._latest_odom
        dx_world = capture_odom.position.x - cur.position.x
        dy_world = capture_odom.position.y - cur.position.y

        # Current yaw (from quaternion → Euler)
        yaw = cur.orientation.to_euler().z

        # Rotate the world-frame displacement into the robot's body frame
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        forward = dx_world * cos_yaw + dy_world * sin_yaw
        left = -dx_world * sin_yaw + dy_world * cos_yaw

        # Heading difference toward the capture pose
        target_yaw = math.atan2(dy_world, dx_world)
        delta_deg = math.degrees(target_yaw - yaw)
        # Normalise to [-180, 180]
        delta_deg = (delta_deg + 180) % 360 - 180

        logger.info(
            "[VLN] Navigating back to capture pose: "
            "forward=%.2f left=%.2f rotate=%.1f deg",
            forward, left, delta_deg,
        )

        try:
            move_rpc = self.get_rpc_calls("UnitreeSkillContainer.relative_move")
            move_rpc(forward, left, delta_deg)
            time.sleep(1.0)  # let robot settle
        except Exception:
            logger.warning("[VLN] relative_move failed during navigate-back")

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _navigate_to_semantic(self, query: str) -> bool:
        """Try to navigate to a location found in the semantic map."""
        try:
            query_rpc = self.get_rpc_calls("SpatialMemory.query_by_text")
        except Exception:
            return False

        results = query_rpc(query)
        if not results:
            return False

        best = results[0]
        similarity = 1.0 - (best.get("distance") or 1)
        if similarity < self.config.similarity_threshold:
            return False

        metadata = best.get("metadata")
        if not metadata:
            return False
        first = metadata[0]
        pose = PoseStamped(
            position=make_vector3(first.get("pos_x", 0), first.get("pos_y", 0), 0),
            orientation=Quaternion.from_euler(make_vector3(0, 0, first.get("rot_z", 0))),
            frame_id="map",
        )

        try:
            set_goal_rpc = self.get_rpc_calls("ReplanningAStarPlanner.set_goal")
        except Exception:
            return False

        return set_goal_rpc(pose)

    def _wait_for_navigation(self, timeout: float = 30.0) -> bool:
        """Block until navigation finishes or timeout. Returns True if goal reached."""
        try:
            get_state_rpc, is_reached_rpc = self.get_rpc_calls(
                "ReplanningAStarPlanner.get_state", "ReplanningAStarPlanner.is_goal_reached"
            )
        except Exception:
            return False

        start = time.time()
        while time.time() - start < timeout:
            if self._search_stop.is_set():
                return False
            state = get_state_rpc()
            if state == NavigationState.IDLE:
                return is_reached_rpc()
            time.sleep(0.5)
        return False

    def _turn_to_face_target(
        self,
        target_x: float,
        target_y: float,
        timeout: float = 3.0,
    ) -> None:
        """Issue a single in-place turn so the robot faces (target_x, target_y).

        Uses the freshest odometry so the heading is accurate at arrival time,
        not at the stale goal-set time used by the intermediate A* steps.
        """
        try:
            set_goal_rpc, cancel_goal_rpc = self.get_rpc_calls(
                "ReplanningAStarPlanner.set_goal",
                "ReplanningAStarPlanner.cancel_goal",
            )
        except Exception:
            return

        odom = self._latest_odom
        if odom is None:
            return

        face_yaw = math.atan2(target_y - odom.position.y, target_x - odom.position.x)
        logger.info(
            "[VLN] Turning to face object: yaw=%.1f° target=(%.2f, %.2f)",
            math.degrees(face_yaw), target_x, target_y,
        )
        goal = PoseStamped(
            position=make_vector3(odom.position.x, odom.position.y, 0),
            orientation=Quaternion.from_euler(make_vector3(0, 0, face_yaw)),
            frame_id="map",
        )
        set_goal_rpc(goal)
        self._wait_for_navigation(timeout=timeout)
        try:
            cancel_goal_rpc()
        except Exception:
            pass

    def _start_exploration(self) -> bool:
        if self.config.exploration_mode == "navdp":
            return self._start_exploration_navdp()
        return self._start_exploration_astar()

    def _stop_exploration(self) -> bool:
        if self.config.exploration_mode == "navdp":
            return self._stop_exploration_navdp()
        return self._stop_exploration_astar()

    # -- A* (WavefrontFrontier) exploration --

    def _start_exploration_astar(self) -> bool:
        try:
            explore_rpc = self.get_rpc_calls("WavefrontFrontierExplorer.explore")
            return explore_rpc()
        except Exception:
            logger.warning("WavefrontFrontierExplorer not connected")
            return False

    def _stop_exploration_astar(self) -> bool:
        try:
            stop_rpc = self.get_rpc_calls("WavefrontFrontierExplorer.stop_exploration")
            return stop_rpc()
        except Exception:
            return False

    # -- NavDP nogoal exploration --

    def _start_exploration_navdp(self) -> bool:
        """Start exploration using NavDP nogoal diffusion policy.

        Sets a dummy language goal so NavDP enters SEEK state and runs
        nogoal_step each tick, generating exploration trajectories purely
        from the diffusion policy.
        """
        try:
            set_lang_rpc = self.get_rpc_calls("NavDPNavigator.set_language_goal")
            set_lang_rpc("explore the environment")
            logger.info("[VLN] NavDP nogoal exploration started")
            return True
        except Exception:
            logger.warning("[VLN] NavDPNavigator not connected, falling back to A*")
            return self._start_exploration_astar()

    def _stop_exploration_navdp(self) -> bool:
        """Stop NavDP nogoal exploration."""
        try:
            cancel_rpc = self.get_rpc_calls("NavDPNavigator.cancel_goal")
            cancel_rpc()
            logger.info("[VLN] NavDP nogoal exploration stopped")
            return True
        except Exception:
            return False

    def _has_navdp(self) -> bool:
        """Check if NavDPNavigator is available and active."""
        try:
            state_rpc = self.get_rpc_calls("NavDPNavigator.get_navdp_state")
            state = state_rpc()
            return state != "UNINITIALIZED"
        except Exception:
            return False

    def _save_navdp_imagegoal_debug(
        self,
        ref_bgr: Any,
        object_description: str,
        *,
        cropped: bool,
    ) -> None:
        """Write the image passed to NavDP imagegoal to disk when configured."""
        root = (self.config.navdp_imagegoal_debug_dir or "").strip()
        if not root:
            return
        try:
            import cv2

            d = Path(root).expanduser()
            d.mkdir(parents=True, exist_ok=True)
            safe = re.sub(r"[^a-zA-Z0-9._-]+", "_", object_description.strip())[:80] or "object"
            ts = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            suffix = "crop" if cropped else "full"
            out = d / f"imagegoal_{ts}_{suffix}_{safe}.png"
            if not cv2.imwrite(str(out), ref_bgr):
                logger.warning("[VLN] Failed to write NavDP imagegoal debug image to %s", out)
            else:
                logger.info("[VLN] Saved NavDP imagegoal reference (%s) → %s", suffix, out)
        except Exception as e:
            logger.warning("[VLN] NavDP imagegoal debug save failed: %s", e)

    def _approach_via_navdp(
        self, object_description: str, bbox: BBox | None = None
    ) -> str:
        """Approach using NavDP imagegoal mode.

        Sends a cropped reference image (centred on the detected bounding box)
        to NavDP, which transitions to APPROACH state and uses imagegoal_step
        to navigate toward the object.  If no bbox is provided the full frame
        is used as a fallback.
        """
        import cv2
        import numpy as np

        if self._latest_image is None:
            return "Error: no image available for NavDP approach."

        # Ensure A* planner is fully stopped before NavDP takes cmd_vel
        try:
            cancel_rpc = self.get_rpc_calls("ReplanningAStarPlanner.cancel_goal")
            cancel_rpc()
        except Exception:
            pass

        # Ensure NavDP motion is unpaused — confirmation VLM checks leave
        # _motion_paused=True (pause_motion is called but resume is skipped for
        # non-resume checks).  Without this, _tick() returns Twist() immediately
        # and the robot never moves during approach.
        # #region agent log
        try:
            import json as _json
            with open("/home/adamliao/work/dimos/.cursor/debug.log", "a") as _dbgf:
                _dbgf.write(_json.dumps({"timestamp": int(time.time()*1000), "hypothesisId": "C-paused", "location": "vln_skill.py:approach_before_resume", "message": "before resume_motion in approach"}) + "\n")
        except Exception:
            pass
        # #endregion
        try:
            navdp_resume = self.get_rpc_calls("NavDPNavigator.resume_motion")
            navdp_resume()
        except Exception:
            pass
        # #region agent log
        try:
            import json as _json
            with open("/home/adamliao/work/dimos/.cursor/debug.log", "a") as _dbgf:
                _dbgf.write(_json.dumps({"timestamp": int(time.time()*1000), "hypothesisId": "C-paused", "location": "vln_skill.py:approach_after_resume", "message": "after resume_motion in approach"}) + "\n")
        except Exception:
            pass
        # #endregion

        # Get reference image (current frame where the object was detected)
        ref_bgr = self._latest_image.data
        from dimos.msgs.sensor_msgs.Image import ImageFormat
        if self._latest_image.format == ImageFormat.RGB:
            ref_bgr = cv2.cvtColor(ref_bgr, cv2.COLOR_RGB2BGR)

        imagegoal_was_cropped = False

        # Crop to bounding box with padding so NavDP gets a focused reference
        if bbox is not None:
            img_h, img_w = ref_bgr.shape[:2]
            x1, y1, x2, y2 = bbox
            # Normalise if coords are in 0-1000 scale
            if x2 > img_w or y2 > img_h:
                x1 = int(x1 / 1000.0 * img_w)
                y1 = int(y1 / 1000.0 * img_h)
                x2 = int(x2 / 1000.0 * img_w)
                y2 = int(y2 / 1000.0 * img_h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # Add 20% padding around the crop
            bw, bh = x2 - x1, y2 - y1
            pad_x, pad_y = int(bw * 0.2), int(bh * 0.2)
            x1 = max(0, x1 - pad_x)
            y1 = max(0, y1 - pad_y)
            x2 = min(img_w, x2 + pad_x)
            y2 = min(img_h, y2 + pad_y)
            if x2 > x1 and y2 > y1:
                ref_bgr = ref_bgr[y1:y2, x1:x2].copy()
                imagegoal_was_cropped = True
                logger.info(
                    "[VLN] Cropped reference image to bbox [%d,%d,%d,%d] "
                    "(padded) → %dx%d",
                    x1, y1, x2, y2, ref_bgr.shape[1], ref_bgr.shape[0],
                )
                # Resize crop to the NavDP inference camera resolution so the
                # policy's letterboxing matches the live observation.
                goal_w = self.config.navdp_imagegoal_width
                goal_h = self.config.navdp_imagegoal_height
                ref_bgr = cv2.resize(ref_bgr, (goal_w, goal_h), interpolation=cv2.INTER_LINEAR)
                logger.info(
                    "[VLN] Resized imagegoal to %dx%d for NavDP", goal_w, goal_h
                )
            else:
                logger.warning("[VLN] Invalid bbox after normalisation, using full frame")

        self._save_navdp_imagegoal_debug(
            ref_bgr, object_description, cropped=imagegoal_was_cropped
        )

        # Set language goal FIRST (resets state machine to SEEK), then
        # set_reference_image SECOND (overrides to APPROACH + imagegoal).
        # Reversed order would leave sm_state=SEEK during imagegoal approach,
        # causing the exploration cost to penalize trajectories toward the target.
        try:
            set_lang_rpc = self.get_rpc_calls("NavDPNavigator.set_language_goal")
            set_lang_rpc(object_description)
        except Exception:
            pass

        try:
            set_ref_rpc = self.get_rpc_calls("NavDPNavigator.set_reference_image")
            set_ref_rpc(ref_bgr)
        except Exception:
            logger.warning("[VLN] Failed to set NavDP reference image, falling back to A*")
            return ""  # empty = signal to fall back

        # #region agent log
        try:
            import json as _json
            _navdp_state = "unknown"
            try:
                _state_rpc = self.get_rpc_calls("NavDPNavigator.get_navdp_state")
                _navdp_state = _state_rpc()
            except Exception:
                pass
            with open("/home/adamliao/work/dimos/.cursor/debug.log", "a") as _dbgf:
                _dbgf.write(_json.dumps({
                    "timestamp": int(time.time() * 1000),
                    "runId": "post-fix-v2",
                    "hypothesisId": "H1-order-fix-verify",
                    "location": "vln_skill.py:approach_after_both_rpcs",
                    "message": "NavDP state after set_language_goal + set_reference_image",
                    "data": {"navdp_state": _navdp_state, "object": object_description},
                }) + "\n")
        except Exception:
            pass
        # #endregion

        logger.info("[VLN] NavDP APPROACH started for '%s'", object_description)

        # Wait for NavDP to reach STOPPED or timeout, periodically running
        # VLM checks (with pause/resume) so the trajectory selector can
        # bias toward the detected object.
        start_time = time.time()
        last_vlm_check = 0.0
        while time.time() - start_time < self.config.approach_timeout:
            if self._search_stop.is_set():
                self._clear_navdp_object_direction()
                try:
                    cancel_rpc = self.get_rpc_calls("NavDPNavigator.cancel_goal")
                    cancel_rpc()
                except Exception:
                    pass
                return "Search was cancelled."

            try:
                state_rpc = self.get_rpc_calls("NavDPNavigator.get_navdp_state")
                navdp_state = state_rpc()
            except Exception:
                navdp_state = "unknown"

            if navdp_state == "STOPPED":
                self._clear_navdp_object_direction()
                logger.info("[VLN] NavDP APPROACH complete → STOPPED")
                return "Successfully approached the object via NavDP imagegoal."

            if navdp_state == "IDLE":
                self._clear_navdp_object_direction()
                logger.info("[VLN] NavDP returned to IDLE during approach")
                return "NavDP approach ended (goal may have been reached)."

            # Periodic VLM check: pause NavDP, run VLM, classify direction,
            # then resume — same pattern as SEEK exploration checks.
            now = time.time()
            if now - last_vlm_check >= self.config.vlm_check_interval:
                last_vlm_check = now
                bbox, _ = self._check_object_in_view(object_description)
                if bbox is not None:
                    self._update_navdp_object_direction(bbox)
                else:
                    self._clear_navdp_object_direction()
                self._resume_motion_after_vlm_check()

            time.sleep(0.5)

        # Timeout — cancel NavDP
        self._clear_navdp_object_direction()
        try:
            cancel_rpc = self.get_rpc_calls("NavDPNavigator.cancel_goal")
            cancel_rpc()
        except Exception:
            pass
        return "NavDP approach timed out. The robot is near the object."

    def _record_instance_to_memory(
        self,
        label: str,
        x: float,
        y: float,
        z: float,
        confidence: float,
        depth_m: float,
        ema_alpha: float = 0.3,
    ) -> None:
        """Record a 3D object instance to NavDPMemory if available (best-effort)."""
        try:
            record_rpc = self.get_rpc_calls("NavDPMemory.record_object_instance")
            record_rpc(
                label=label,
                x=x, y=y, z=z,
                confidence=confidence,
                depth_m=depth_m,
                ema_alpha=ema_alpha,
            )
        except Exception:
            pass  # NavDPMemory is optional

    def _bbox_bearing_yaw(self, bbox: BBox, image_w: int, current_yaw: float) -> float:
        """Compute world-frame heading that keeps the bbox centre in the camera FOV.

        The bbox centre pixel offset from the image centre is converted to a yaw
        correction and added to the robot's current heading.  The correction is
        capped at ±30° so the robot turns gradually, not in a single lurch.

        Args:
            bbox: Bounding box (x1, y1, x2, y2) in pixel or 0-1000 scale.
            image_w: Width of the camera image in pixels.
            current_yaw: Robot's current heading in radians.

        Returns:
            New world-frame yaw to face the bbox centre.
        """
        bbox_cx = (bbox[0] + bbox[2]) / 2.0
        # Normalise 0-1000 → pixel if needed
        if bbox[2] > image_w:
            bbox_cx = bbox_cx / 1000.0 * image_w
        # Fractional offset: -1 (far left) … +1 (far right)
        offset_ratio = (bbox_cx - image_w / 2.0) / (image_w / 2.0)
        # Camera horizontal FOV ≈ 2*atan(img_w/(2*fx)); clamp turn to ±30°
        correction = offset_ratio * math.radians(30.0)
        return current_yaw + correction

    def _approach_via_astar(self, bbox: BBox, object_description: str = "") -> str:
        """Approach a detected object using A* navigation with 3D depth-based targeting.

        Strategy: "face-then-step" — each A* goal carries the yaw that keeps the
        detected object in the camera FOV.  The yaw is derived from:
          1. The bbox pixel bearing (offset from image centre → heading correction).
          2. Blended with the 3D target bearing once the estimate stabilises.
        This avoids the robot spinning away from the object between steps.

        The target 3D position is updated after each step with full replacement
        (alpha=1) for the first 3 observations to avoid anchoring on a noisy
        initial estimate, then EMA smoothing kicks in.

        Each run saves a JSON trajectory file to ``assets/output/`` for debugging.
        """
        try:
            set_goal_rpc, cancel_goal_rpc = self.get_rpc_calls(
                "ReplanningAStarPlanner.set_goal",
                "ReplanningAStarPlanner.cancel_goal",
            )
        except Exception:
            return "Error: NavigationInterface not connected for approach."

        if self._latest_odom is None:
            return "Error: no odometry available for approach."

        try:
            cancel_goal_rpc()
        except Exception:
            pass

        stop_distance = self.config.approach_stop_distance
        min_confidence = self.config.approach_min_depth_confidence
        ema_alpha = self.config.approach_ema_alpha
        max_steps = max(1, int(self.config.approach_timeout / 4))  # ~4s per step
        max_lost_retries = 3
        lost_retries = 0

        from dimos.navigation.visual.object_localizer import (
            ObjectEstimate,
            ObjectInstance,
            ObjectLocalizer,
        )

        # ── Trajectory log (Task 2) ──────────────────────────────────────────
        approach_log: list[dict] = []
        _ts = time.strftime("%Y%m%d_%H%M%S")
        _obj_slug = re.sub(r"[^a-z0-9]+", "_", object_description.lower())[:30]
        _log_path = Path("assets/output") / f"approach_{_ts}_{_obj_slug}.json"

        def _log_step(
            step_idx: int,
            robot_x: float, robot_y: float, robot_yaw: float,
            goal_x: float, goal_y: float, goal_yaw: float,
            target_x: float | None, target_y: float | None,
            depth_m: float | None,
            bbox_raw: tuple | None,
            found: bool,
            event: str = "",
        ) -> None:
            approach_log.append({
                "step": step_idx,
                "robot": {"x": round(robot_x, 3), "y": round(robot_y, 3), "yaw_deg": round(math.degrees(robot_yaw), 1)},
                "goal": {"x": round(goal_x, 3), "y": round(goal_y, 3), "yaw_deg": round(math.degrees(goal_yaw), 1)},
                "target": {"x": round(target_x, 3), "y": round(target_y, 3)} if target_x is not None else None,
                "depth_m": round(depth_m, 2) if depth_m is not None else None,
                "bbox": list(bbox_raw) if bbox_raw else None,
                "found": found,
                "event": event,
            })

        def _save_log(outcome: str) -> None:
            try:
                _log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(_log_path, "w") as f:
                    json.dump({"object": object_description, "outcome": outcome, "steps": approach_log}, f, indent=2)
                logger.info("[VLN] Approach trajectory saved: %s (%d steps)", _log_path, len(approach_log))
            except Exception as e:
                logger.warning("[VLN] Could not save approach log: %s", e)

        # ── Initial 3D estimate ──────────────────────────────────────────────
        target: ObjectEstimate | None = None
        instance: ObjectInstance | None = None

        if (
            self._localizer is not None
            and self._latest_depth is not None
        ):
            target = self._localizer.estimate_from_bbox_depth(
                bbox, self._latest_depth, self._latest_odom,
                min_confidence=min_confidence,
            )
            if target is not None:
                instance = ObjectInstance(label=object_description, first_seen_odom=self._latest_odom)
                instance.update(target, alpha=1.0)
                logger.info(
                    "[VLN] Initial 3D estimate: (%.2f, %.2f, %.2f) depth=%.2fm conf=%.2f",
                    target.x, target.y, target.z, target.depth_m, target.confidence,
                )
                self._record_instance_to_memory(
                    label=object_description,
                    x=target.x, y=target.y, z=target.z,
                    confidence=target.confidence,
                    depth_m=target.depth_m,
                    ema_alpha=1.0,
                )

        # ── Approach loop ────────────────────────────────────────────────────
        # FAR mode: depth > 2.5 m or dist > 3.0 m → step forward along bbox bearing.
        # NEAR mode: depth ≤ 2.5 m and dist ≤ 3.0 m → follow 3D target line.
        # Yaw is clamped to ±15° per step to avoid sudden spins.
        _FAR_DEPTH_M   = 2.5
        _FAR_DIST_M    = 3.0
        _FAR_STEP_M    = 0.6   # metres to advance per FAR step
        _MAX_YAW_DELTA = math.radians(15.0)

        prev_goal_yaw: float | None = None

        for step in range(max_steps):
            if self._search_stop.is_set():
                _save_log("cancelled")
                return "Search was cancelled."

            odom = self._latest_odom
            if odom is None:
                time.sleep(0.5)
                continue

            # Current robot yaw from quaternion
            q = odom.orientation
            cur_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )

            img_w = self._latest_image.data.shape[1] if self._latest_image is not None else 640
            bbox_yaw = self._bbox_bearing_yaw(bbox, img_w, cur_yaw)

            dist_to_target = math.hypot(
                target.x - odom.position.x, target.y - odom.position.y
            ) if target is not None else float("inf")

            # Determine phase
            far_mode = (
                target is None
                or target.depth_m > _FAR_DEPTH_M
                or dist_to_target > _FAR_DIST_M
            )

            if far_mode:
                # Walk a fixed step along the (smoothed) bbox bearing.
                # The 3D target is still updated in the background for when NEAR kicks in.
                raw_yaw = bbox_yaw
                if prev_goal_yaw is not None:
                    delta = (raw_yaw - prev_goal_yaw + math.pi) % (2 * math.pi) - math.pi
                    delta = max(-_MAX_YAW_DELTA, min(_MAX_YAW_DELTA, delta))
                    goal_yaw = prev_goal_yaw + delta
                else:
                    goal_yaw = raw_yaw
                goal_x = odom.position.x + _FAR_STEP_M * math.cos(goal_yaw)
                goal_y = odom.position.y + _FAR_STEP_M * math.sin(goal_yaw)
                mode_tag = "far"
                logger.info(
                    "[VLN] A* approach step %d/%d (FAR): depth=%.2fm dist=%.2fm "
                    "goal=(%.2f,%.2f) yaw=%.1f°",
                    step + 1, max_steps,
                    target.depth_m if target else -1.0, dist_to_target,
                    goal_x, goal_y, math.degrees(goal_yaw),
                )
            else:
                # NEAR mode: accurate 3D estimate — follow robot→target line.
                goal_x, goal_y, target_bearing = ObjectLocalizer.compute_approach_goal(
                    odom, target, stop_distance
                )
                raw_yaw = 0.4 * bbox_yaw + 0.6 * target_bearing
                if prev_goal_yaw is not None:
                    delta = (raw_yaw - prev_goal_yaw + math.pi) % (2 * math.pi) - math.pi
                    delta = max(-_MAX_YAW_DELTA, min(_MAX_YAW_DELTA, delta))
                    goal_yaw = prev_goal_yaw + delta
                else:
                    goal_yaw = raw_yaw
                mode_tag = "near"
                logger.info(
                    "[VLN] A* approach step %d/%d (NEAR): target=(%.2f,%.2f) "
                    "dist=%.2fm goal=(%.2f,%.2f) yaw=%.1f° bbox_yaw=%.1f° target_yaw=%.1f°",
                    step + 1, max_steps,
                    target.x, target.y, dist_to_target,
                    goal_x, goal_y, math.degrees(goal_yaw),
                    math.degrees(bbox_yaw), math.degrees(target_bearing),
                )

                if dist_to_target <= stop_distance:
                    logger.info("[VLN] Reached stop distance (%.2fm)", dist_to_target)
                    try:
                        cancel_goal_rpc()
                    except Exception:
                        pass
                    self._turn_to_face_target(target.x, target.y)
                    _save_log("success_stop_distance")
                    return "Successfully approached the object."

            prev_goal_yaw = goal_yaw

            _log_step(step + 1, odom.position.x, odom.position.y, cur_yaw,
                      goal_x, goal_y, goal_yaw,
                      target.x if target else None, target.y if target else None,
                      target.depth_m if target else None, bbox, True, mode_tag)

            # Send the goal with the yaw so the A* planner's final rotation
            # leaves the robot facing the object (keeps object in FOV for VLM check).
            goal = PoseStamped(
                position=make_vector3(goal_x, goal_y, 0),
                orientation=Quaternion.from_euler(make_vector3(0, 0, goal_yaw)),
                frame_id="map",
            )
            set_goal_rpc(goal)
            self._wait_for_navigation(timeout=4.0)

            # Re-detect and update 3D estimate
            new_bbox, _ = self._check_object_in_view(object_description)
            if new_bbox:
                bbox = new_bbox
                lost_retries = 0

                if (
                    self._localizer is not None
                    and self._latest_depth is not None
                    and self._latest_odom is not None
                ):
                    new_est = self._localizer.estimate_from_bbox_depth(
                        new_bbox, self._latest_depth, self._latest_odom,
                        min_confidence=min_confidence,
                    )
                    if new_est is not None:
                        if target is None:
                            target = new_est
                            instance = ObjectInstance(
                                label=object_description,
                                first_seen_odom=self._latest_odom,
                            )
                            instance.update(new_est, alpha=1.0)
                        else:
                            # Use full replacement (alpha=1) for the first
                            # few observations so a bad initial estimate
                            # does not persist via EMA smoothing.
                            obs_count = instance.observation_count if instance else 0
                            alpha = 1.0 if obs_count < 3 else ema_alpha
                            target = ObjectEstimate(
                                x=alpha * new_est.x + (1.0 - alpha) * target.x,
                                y=alpha * new_est.y + (1.0 - alpha) * target.y,
                                z=alpha * new_est.z + (1.0 - alpha) * target.z,
                                confidence=new_est.confidence,
                                depth_m=new_est.depth_m,
                                method=new_est.method,
                            )
                            if instance is not None:
                                instance.update(new_est, alpha=alpha)
                        logger.info(
                            "[VLN] Updated 3D target: (%.2f, %.2f, %.2f) depth=%.2fm",
                            target.x, target.y, target.z, target.depth_m,
                        )
                        # Persist to NavDPMemory (best-effort, non-blocking)
                        self._record_instance_to_memory(
                            label=object_description,
                            x=target.x, y=target.y, z=target.z,
                            confidence=target.confidence,
                            depth_m=target.depth_m,
                            ema_alpha=ema_alpha,
                        )

                # Pixel-fill stop check (covers cases where depth isn't available)
                if self._latest_image is not None:
                    img_h, img_w = self._latest_image.data.shape[:2]
                    area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
                    if new_bbox[2] > img_w or new_bbox[3] > img_h:
                        area = (
                            (new_bbox[2] - new_bbox[0]) / 1000.0 * img_w
                            * (new_bbox[3] - new_bbox[1]) / 1000.0 * img_h
                        )
                    fill_ratio = area / (img_w * img_h)
                    logger.info("[VLN] Object fill_ratio=%.2f", fill_ratio)
                    if fill_ratio > 0.30:
                        try:
                            cancel_goal_rpc()
                        except Exception:
                            pass
                        if target is not None:
                            self._turn_to_face_target(target.x, target.y)
                        _save_log("success_fill_ratio")
                        return "Successfully approached the object."
            else:
                lost_retries += 1
                logger.info(
                    "[VLN] Lost object from view during approach (attempt %d/%d)",
                    lost_retries, max_lost_retries,
                )
                odom_lost = self._latest_odom
                _log_step(step + 1, odom_lost.position.x if odom_lost else 0,
                          odom_lost.position.y if odom_lost else 0, cur_yaw,
                          goal_x, goal_y, goal_yaw,
                          target.x if target else None, target.y if target else None,
                          target.depth_m if target else None, None, False, "lost")
                if lost_retries >= max_lost_retries:
                    try:
                        cancel_goal_rpc()
                    except Exception:
                        pass
                    _save_log("lost_max_retries")
                    return (
                        "Object was visible but lost during approach. "
                        "The robot is near the last sighting."
                    )
                time.sleep(1.0)

        try:
            cancel_goal_rpc()
        except Exception:
            pass
        if target is not None:
            self._turn_to_face_target(target.x, target.y)
        _save_log("max_steps_reached")
        return "Approach complete (max steps reached). The robot is near the object."

    def _confirm_detection(self, obj: str) -> "BBox | None":
        """Confirm a VLM detection with a majority vote at the current heading.

        Runs VLM ``confirm_checks`` times at the robot's current heading
        (where the object was just detected) and requires at least
        ``confirm_threshold`` positives.

        Returns the best confirmed bbox, or None if not confirmed.
        """
        checks = self.config.confirm_checks
        threshold = self.config.confirm_threshold
        check_delay = self.config.confirm_check_delay

        heading_deg: float | str = "unknown"
        if self._latest_odom is not None:
            heading_deg = round(
                math.degrees(self._latest_odom.orientation.to_euler().z), 1
            )
        logger.info(
            "[VLN] Confirming '%s' at current heading (yaw=%s), threshold=%d/%d",
            obj, heading_deg, threshold, checks,
        )

        detections = 0
        best_bbox = None
        for check_idx in range(checks):
            if self._search_stop.is_set():
                return None
            time.sleep(check_delay)
            bbox, _ = self._check_object_in_view(obj)
            found = bbox is not None
            if found:
                detections += 1
                best_bbox = bbox
            logger.info(
                "[VLN] Confirm check %d/%d for '%s': %s",
                check_idx + 1, checks, obj,
                f"FOUND bbox={bbox}" if found else "not found",
            )

        if detections >= threshold:
            logger.info(
                "[VLN] Object '%s' confirmed (%d/%d checks passed)",
                obj, detections, checks,
            )
            return best_bbox

        logger.info(
            "[VLN] Object '%s' not confirmed (%d/%d checks passed, need %d)",
            obj, detections, checks, threshold,
        )
        return None

    def _approach_object(self, bbox: BBox, object_description: str = "") -> str:
        """Approach a detected object. Uses NavDP imagegoal if available, else A*.

        When NavDP is active, a cropped reference image (centred on the bbox)
        is sent to NavDP, and the diffusion policy drives toward it using
        imagegoal_step. This is more robust than A* step-by-step approach
        because the policy handles obstacle avoidance and visual servoing
        natively.
        """
        # Try NavDP imagegoal approach first
        if self._has_navdp():
            logger.info("[VLN] Using NavDP imagegoal for approach (bbox=%s)", bbox)
            result = self._approach_via_navdp(object_description, bbox=bbox)
            if result:  # non-empty = NavDP handled it
                return result
            logger.info("[VLN] NavDP approach failed, falling back to A*")

        # Fallback to A* step-by-step approach
        return self._approach_via_astar(bbox, object_description)

    # ------------------------------------------------------------------
    # Cancel helper
    # ------------------------------------------------------------------

    def _cancel_search(self) -> None:
        if self._search_active:
            self._search_stop.set()
            self._stop_exploration()
            try:
                cancel_rpc = self.get_rpc_calls("ReplanningAStarPlanner.cancel_goal")
                cancel_rpc()
            except Exception:
                pass
            # Also cancel NavDP goal if active
            try:
                navdp_cancel = self.get_rpc_calls("NavDPNavigator.cancel_goal")
                navdp_cancel()
            except Exception:
                pass
            if (
                self._search_thread
                and self._search_thread.is_alive()
                and threading.current_thread() != self._search_thread
            ):
                self._search_thread.join(timeout=3.0)
            self._search_active = False

    # ------------------------------------------------------------------
    # Core VLN search loop
    # ------------------------------------------------------------------

    def _vln_search(self, room: str | None, obj: str | None) -> str:
        """Execute the full VLN pipeline. Runs in a worker thread."""
        # Phase 1 — Navigate to room (if specified)
        if room:
            logger.info(f"[VLN] Phase 1: navigating to room '{room}'")

            # First, try the semantic map
            if self._navigate_to_semantic(room):
                logger.info(f"[VLN] Found '{room}' in semantic map, navigating...")
                reached = self._wait_for_navigation(timeout=60.0)
                if reached:
                    logger.info(f"[VLN] Reached semantic map location for '{room}'")
                else:
                    logger.info(f"[VLN] Could not reach semantic map location for '{room}'")

            # If not in semantic map or didn't reach, explore and look for the room
            if not self._check_room_match(room):
                logger.info(f"[VLN] Not in '{room}' yet, starting exploration to find it")
                self._start_exploration()

                search_start = time.time()
                while time.time() - search_start < self.config.search_timeout / 2:
                    if self._search_stop.is_set():
                        self._stop_exploration()
                        return "Search cancelled."

                    if self._check_room_match(room, resume_search=True):
                        logger.info(f"[VLN] VLM confirmed arrival at '{room}'")
                        self._stop_exploration()
                        # Tag this location for future reference
                        try:
                            tag_rpc = self.get_rpc_calls("SpatialMemory.tag_location")
                            from dimos.types.robot_location import RobotLocation

                            if self._latest_odom:
                                pos = self._latest_odom.position
                                tag_rpc(
                                    RobotLocation(
                                        name=room,
                                        position=(pos.x, pos.y, pos.z),
                                        rotation=(0, 0, 0),
                                    )
                                )
                        except Exception:
                            pass
                        break

                    time.sleep(self.config.vlm_check_interval)
                else:
                    self._stop_exploration()
                    logger.info(f"[VLN] Could not find room '{room}' within timeout")
                    # Continue anyway — object may still be findable
            else:
                logger.info(f"[VLN] Already in '{room}'")

        # Phase 2 — If no object specified, we're done
        if not obj:
            return f"Navigated to '{room}'." if room else "No goal specified."

        # Phase 3 — Active object search
        logger.info(f"[VLN] Phase 2: searching for object '{obj}'")

        # First: check if object is already in view
        bbox, _ = self._check_object_in_view(obj)
        if bbox:
            logger.info(f"[VLN] Object '{obj}' already in view!")
            result = self._approach_object(bbox, obj)
            return f"Found '{obj}'. {result}"

        # Second: check semantic memory
        if self._navigate_to_semantic(obj):
            logger.info(f"[VLN] Found '{obj}' in semantic map, navigating...")
            reached = self._wait_for_navigation(timeout=30.0)
            if reached:
                # Double-check with VLM at destination
                bbox, _ = self._check_object_in_view(obj)
                if bbox:
                    result = self._approach_object(bbox, obj)
                    return f"Found '{obj}' via semantic map. {result}"
                return f"Reached semantic map location for '{obj}' but could not visually confirm it."

        # Third: explore while continuously checking VLM
        # Adaptive timeout: keeps exploring as long as coverage is growing.
        # Terminates when (a) object found, (b) hard timeout, (c) stall
        # timeout (no new coverage for N seconds), or (d) the costmap
        # boundary is fully observed.
        logger.info(f"[VLN] Starting exploration with active VLM search for '{obj}'")
        self._start_exploration()

        search_start = time.time()
        last_progress_time = search_start
        last_trail_distance = 0.0
        last_observed_fraction = 0.0

        # Try to get initial exploration stats
        try:
            stats_rpc = self.get_rpc_calls("NavDPNavigator.get_exploration_stats")
            init_stats = stats_rpc()
            last_trail_distance = init_stats.get("trail_distance_m", 0.0)
            last_observed_fraction = init_stats.get("observed_fraction", 0.0)
        except Exception:
            stats_rpc = None

        while True:
            now = time.time()
            elapsed = now - search_start
            stalled_for = now - last_progress_time

            # --- Termination conditions ---
            # (a) Hard timeout
            if elapsed > self.config.search_timeout:
                logger.info(
                    "[VLN] Search hard timeout (%.0fs > %.0fs)",
                    elapsed, self.config.search_timeout,
                )
                break

            # (b) Stall timeout — robot isn't covering new ground
            if stalled_for > self.config.search_stall_timeout:
                logger.info(
                    "[VLN] Search stall timeout: no new coverage for %.0fs "
                    "(threshold %.0fs)",
                    stalled_for, self.config.search_stall_timeout,
                )
                break

            # (c) Cancelled externally
            if self._search_stop.is_set():
                self._stop_exploration()
                return "Search cancelled."

            # --- VLM check for the target object ---
            bbox, capture_odom = self._check_object_in_view(obj, resume_search=True)
            if bbox:
                logger.info(
                    "[VLN] Found '%s' during exploration! (mode=%s)",
                    obj, self.config.exploration_mode,
                )

                # Confirmation phase: stop, majority-vote VLM, rotate+recheck
                # up to a full 360 degrees before giving up on this detection.
                # 1.5s settle ensures MuJoCo momentum has dissipated and the
                # camera has a stable, non-blurred frame before the first check.
                self._stop_exploration()
                time.sleep(1.5)  # let robot settle after stop
                bbox = self._confirm_detection(obj)
                if bbox:
                    result = self._approach_object(bbox, obj)
                    return f"Found '{obj}' during exploration. {result}"
                else:
                    logger.info("[VLN] Detection not confirmed after 360 scan, resuming exploration")
                    self._start_exploration()
                    # Reset stall timer after a false-positive; confirmation
                    # takes time and should not count as stalling.
                    last_progress_time = time.time()
                    continue

            # --- Exploration progress check (every VLM interval) ---
            if stats_rpc is not None:
                try:
                    stats = stats_rpc()
                    cur_trail_dist = stats.get("trail_distance_m", 0.0)
                    cur_observed = stats.get("observed_fraction", 0.0)

                    # Check progress: did the robot cover new ground?
                    new_distance = cur_trail_dist - last_trail_distance
                    if new_distance >= self.config.search_progress_distance:
                        last_progress_time = time.time()
                        last_trail_distance = cur_trail_dist
                        last_observed_fraction = cur_observed

                    # (d) Boundary fully observed — costmap is mostly known
                    if (
                        stats.get("costmap_available", False)
                        and cur_observed >= self.config.search_observed_fraction
                    ):
                        logger.info(
                            "[VLN] Boundary fully observed: %.1f%% of costmap known "
                            "(threshold %.0f%%). Stopping search.",
                            cur_observed * 100,
                            self.config.search_observed_fraction * 100,
                        )
                        break
                except Exception:
                    pass  # stats RPC unavailable — ignore

            time.sleep(self.config.vlm_check_interval)

        self._stop_exploration()
        # Build an informative message for the agent
        trail_dist_str = f"{last_trail_distance:.1f}m explored"
        obs_str = f"{last_observed_fraction * 100:.0f}% of map observed"
        return (
            f"Could not find '{obj}' after {elapsed:.0f}s. "
            f"{trail_dist_str}, {obs_str}. "
            f"The robot remembers where it has been — retrying may cover new areas."
        )

    # ------------------------------------------------------------------
    # Skills exposed to the agent
    # ------------------------------------------------------------------

    @skill
    def find_object_in_room(self, goal: str) -> str:
        """Find an object, optionally within a specific room.

        Decomposes compound goals like "glasses box in the CTO office" into
        a two-phase search: first navigate to the room, then actively search
        for the object by exploring and continuously checking the camera.

        Use this for complex navigation goals that involve both a location
        and a specific object to find.

        Args:
            goal: Natural language description, e.g. "glasses box in the CTO office room"

        Returns:
            str: Result of the search
        """
        if not self._started:
            return "Error: VLN skill has not been started."

        if self._search_active:
            return "A VLN search is already in progress. Cancel it first with stop_vln_search."

        # Decompose the goal
        parts = self._decompose_goal(goal)
        room = parts.get("room")
        obj = parts.get("object")

        logger.info(f"[VLN] Decomposed goal: room={room!r}, object={obj!r}")

        if not room and not obj:
            return "Could not understand the goal. Please rephrase."

        # Run the search in a thread so the skill can report progress
        self._search_active = True
        self._search_stop.clear()
        self._search_result = None

        def _run() -> None:
            try:
                self._search_result = self._vln_search(room, obj)
            except Exception as e:
                logger.error(f"[VLN] Search failed: {e}", exc_info=True)
                self._search_result = f"Search failed with error: {e}"
            finally:
                self._search_active = False

        self._search_thread = threading.Thread(target=_run, daemon=True)
        self._search_thread.start()

        # Wait for the search to complete (blocking skill call)
        self._search_thread.join(timeout=self.config.search_timeout + 30)

        # If the thread is still alive after the join window the RPC caller
        # has likely already timed out.  Force-cancel so NavDP stops moving.
        if self._search_thread.is_alive():
            logger.warning("[VLN] Search thread still alive after join timeout — force-cancelling")
            self._cancel_search()

        result = self._search_result or "Search ended without a result."
        decomp_info = f"[Decomposed: room={room!r}, object={obj!r}]"
        return f"{decomp_info} {result}"

    @skill
    def stop_vln_search(self) -> str:
        """Cancel the current VLN search. The robot will stop moving."""
        if not self._search_active:
            return "No VLN search is currently active."
        self._cancel_search()
        return "VLN search cancelled. The robot has stopped."

    @skill
    def active_search(self, object_description: str) -> str:
        """Explore the environment while actively looking for a specific object.

        Unlike navigate_with_text which checks the camera only once, this skill
        continuously checks the camera feed while the robot explores via
        frontier exploration. Use this when you need to find an object that
        is not currently visible and not in the semantic memory.

        Args:
            object_description: What to look for, e.g. "red backpack", "glasses box"

        Returns:
            str: Result of the search
        """
        if not self._started:
            return "Error: VLN skill has not been started."

        if self._search_active:
            return "A search is already in progress. Cancel it first with stop_vln_search."

        self._search_active = True
        self._search_stop.clear()
        self._search_result = None

        def _run() -> None:
            try:
                self._search_result = self._vln_search(None, object_description)
            except Exception as e:
                logger.error(f"[VLN] Search failed: {e}", exc_info=True)
                self._search_result = f"Search failed with error: {e}"
            finally:
                self._search_active = False

        self._search_thread = threading.Thread(target=_run, daemon=True)
        self._search_thread.start()
        self._search_thread.join(timeout=self.config.search_timeout + 30)

        # Force-cancel if thread outlived the join window to prevent NavDP
        # from continuing to drive unsupervised after an RPC timeout.
        if self._search_thread.is_alive():
            logger.warning("[VLN] Search thread still alive after join timeout — force-cancelling")
            self._cancel_search()

        return self._search_result or "Search ended without a result."


__all__ = ["VLNSkillContainer"]

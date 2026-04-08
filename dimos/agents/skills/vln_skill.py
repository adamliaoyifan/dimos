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

import math
import threading
import time
from typing import Any

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


class VLNConfig(ModuleConfig):
    """Configuration for the VLN skill."""

    vlm_check_interval: float = 2.0
    """Seconds between VLM checks during active search."""

    search_timeout: float = 120.0
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

    confirm_rotate_deg: float = 20.0
    """Degrees to rotate between confirmation checks (full 360 scan if needed).
    Should be less than the camera horizontal FOV (~38° for Go2) to avoid gaps."""

    confirm_check_delay: float = 1.5
    """Seconds to wait between VLM checks during confirmation (ensures fresh camera frame)."""

    max_overrun_m: float = 1.5
    """Maximum distance (metres) the robot may travel past the detection capture pose
    before it navigates back to the capture point before confirmation.
    Set to 0 to always navigate back; set large to skip the back-up."""


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
        "WavefrontFrontierExplorer.explore",
        "WavefrontFrontierExplorer.stop_exploration",
        "WavefrontFrontierExplorer.is_exploration_active",
        "SpatialMemory.query_by_text",
        "SpatialMemory.tag_location",
        "SpatialMemory.query_tagged_location",
        # NavDP (optional — used for imagegoal approach when available)
        "NavDPNavigator.set_language_goal",
        "NavDPNavigator.set_reference_image",
        "NavDPNavigator.get_navdp_state",
        "NavDPNavigator.cancel_goal",
        # In-place rotation during detection confirmation (uses A* planner, collision-safe)
        "UnitreeSkillContainer.relative_move",
    ]

    color_image: In[Image]
    odom: In[PoseStamped]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._vl_model = self._create_vlm()

        self._latest_image: Image | None = None
        self._latest_odom: PoseStamped | None = None
        self._started = False

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
        self._disposables.add(Disposable(self.odom.subscribe(self._on_odom)))
        self._started = True
        logger.info(
            "[VLN] started — exploration_mode=%s, vlm=%s",
            self.config.exploration_mode, self.config.vlm_backend,
        )

    @rpc
    def stop(self) -> None:
        self._cancel_search()
        super().stop()

    def _on_image(self, image: Image) -> None:
        self._latest_image = image

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

    def _check_room_match(self, room_description: str) -> bool:
        """Ask the VLM whether the current camera view matches the room."""
        if self._latest_image is None:
            return False

        prompt = (
            f'Look at this image. Is this a "{room_description}"?\n'
            "Answer with ONLY 'yes' or 'no'."
        )
        response = self._vl_model.query(self._latest_image, prompt)
        return "yes" in response.lower().split()

    # ------------------------------------------------------------------
    # Object detection via VLM
    # ------------------------------------------------------------------

    def _check_object_in_view(
        self, object_description: str
    ) -> tuple["BBox | None", "PoseStamped | None"]:
        """Check if the target object is visible in the current frame.

        Returns (bbox, capture_odom) where capture_odom is the robot pose at
        the moment the image was captured (before the blocking VLM call).
        Callers can compare capture_odom to the current odom to detect overrun.
        """
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
            logger.info("[VLN] VLM check '%s': EXCEPTION after %.0fms — %s", object_description, elapsed_ms, exc)
            return None, capture_odom
        elapsed_ms = (time.time() - t0) * 1000
        # Log current position (at result time) for diagnostics
        _pos = (round(self._latest_odom.position.x, 2), round(self._latest_odom.position.y, 2)) if self._latest_odom else None
        _cap_pos = (round(capture_odom.position.x, 2), round(capture_odom.position.y, 2)) if capture_odom else None
        logger.info(
            "[VLN] VLM check '%s': %s (%.0fms) pos=%s cap_pos=%s",
            object_description,
            f"FOUND bbox={result}" if result else "not found",
            elapsed_ms,
            _pos,
            _cap_pos,
        )
        return result, capture_odom

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

        # Get reference image (current frame where the object was detected)
        ref_bgr = self._latest_image.data
        from dimos.msgs.sensor_msgs.Image import ImageFormat
        if self._latest_image.format == ImageFormat.RGB:
            ref_bgr = cv2.cvtColor(ref_bgr, cv2.COLOR_RGB2BGR)

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
                logger.info(
                    "[VLN] Cropped reference image to bbox [%d,%d,%d,%d] "
                    "(padded) → %dx%d",
                    x1, y1, x2, y2, ref_bgr.shape[1], ref_bgr.shape[0],
                )
            else:
                logger.warning("[VLN] Invalid bbox after normalisation, using full frame")

        try:
            set_ref_rpc = self.get_rpc_calls("NavDPNavigator.set_reference_image")
            set_ref_rpc(ref_bgr)
        except Exception:
            logger.warning("[VLN] Failed to set NavDP reference image, falling back to A*")
            return ""  # empty = signal to fall back

        # Also set the language goal so VLM detection can track
        try:
            set_lang_rpc = self.get_rpc_calls("NavDPNavigator.set_language_goal")
            set_lang_rpc(object_description)
        except Exception:
            pass

        logger.info("[VLN] NavDP APPROACH started for '%s'", object_description)

        # Wait for NavDP to reach STOPPED or timeout
        start_time = time.time()
        while time.time() - start_time < self.config.approach_timeout:
            if self._search_stop.is_set():
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
                logger.info("[VLN] NavDP APPROACH complete → STOPPED")
                return "Successfully approached the object via NavDP imagegoal."

            if navdp_state == "IDLE":
                logger.info("[VLN] NavDP returned to IDLE during approach")
                return "NavDP approach ended (goal may have been reached)."

            time.sleep(0.5)

        # Timeout — cancel NavDP
        try:
            cancel_rpc = self.get_rpc_calls("NavDPNavigator.cancel_goal")
            cancel_rpc()
        except Exception:
            pass
        return "NavDP approach timed out. The robot is near the object."

    def _approach_via_astar(self, bbox: BBox, object_description: str = "") -> str:
        """Fallback approach using A* planner small steps.

        Used when NavDP is not available.
        """
        import math

        try:
            set_goal_rpc, cancel_goal_rpc = self.get_rpc_calls(
                "ReplanningAStarPlanner.set_goal",
                "ReplanningAStarPlanner.cancel_goal",
            )
        except Exception:
            return "Error: NavigationInterface not connected for approach."

        if self._latest_odom is None:
            return "Error: no odometry available for approach."

        # Cancel any existing goal before starting approach
        try:
            cancel_goal_rpc()
        except Exception:
            pass

        step_distance = 0.3  # meters per step (smaller for more control)
        max_steps = int(self.config.approach_timeout / 4)  # ~4s per step
        lost_retries = 0
        max_lost_retries = 3

        for step in range(max_steps):
            if self._search_stop.is_set():
                return "Search was cancelled."

            q = self._latest_odom.orientation
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw = math.atan2(siny, cosy)

            # Steer toward the bbox centre
            if bbox and self._latest_image is not None:
                img_w = self._latest_image.data.shape[1]
                bbox_cx = (bbox[0] + bbox[2]) / 2.0
                # Normalize bbox coords if they appear to be in 0-1000 scale
                if bbox[2] > img_w:
                    bbox_cx = bbox_cx / 1000.0 * img_w
                offset_ratio = (bbox_cx - img_w / 2.0) / (img_w / 2.0)
                yaw += offset_ratio * 0.5

            goal_x = self._latest_odom.position.x + step_distance * math.cos(yaw)
            goal_y = self._latest_odom.position.y + step_distance * math.sin(yaw)

            goal = PoseStamped(
                position=make_vector3(goal_x, goal_y, 0),
                orientation=Quaternion.from_euler(make_vector3(0, 0, yaw)),
                frame_id="map",
            )
            set_goal_rpc(goal)
            logger.info(
                "[VLN] A* approach step %d: moving %.2fm toward (%.2f, %.2f), yaw=%.1f°",
                step + 1, step_distance, goal_x, goal_y, math.degrees(yaw),
            )

            self._wait_for_navigation(timeout=6.0)

            new_bbox, _ = self._check_object_in_view(object_description)
            if new_bbox:
                lost_retries = 0
                new_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])
                bbox = new_bbox
                if self._latest_image is not None:
                    img_h, img_w = self._latest_image.data.shape[:2]
                    # Normalize bbox area if coords are in 0-1000 scale
                    if new_bbox[2] > img_w or new_bbox[3] > img_h:
                        scale_x = img_w / 1000.0
                        scale_y = img_h / 1000.0
                        new_area = (
                            (new_bbox[2] - new_bbox[0]) * scale_x
                            * (new_bbox[3] - new_bbox[1]) * scale_y
                        )
                    fill_ratio = new_area / (img_w * img_h)
                    logger.info("[VLN] Object still in view, fill_ratio=%.2f", fill_ratio)
                    if fill_ratio > 0.30:
                        try:
                            cancel_goal_rpc()
                        except Exception:
                            pass
                        return "Successfully approached the object."
            else:
                lost_retries += 1
                logger.info(
                    "[VLN] Lost object from view during approach (attempt %d/%d)",
                    lost_retries, max_lost_retries,
                )
                if lost_retries >= max_lost_retries:
                    try:
                        cancel_goal_rpc()
                    except Exception:
                        pass
                    return "Object was visible but lost during approach. The robot is near the last sighting."
                time.sleep(1.0)

        try:
            cancel_goal_rpc()
        except Exception:
            pass
        return "Approach complete (max steps reached). The robot is near the object."

    def _confirm_detection(self, obj: str, initial_bbox: "BBox | None" = None) -> "BBox | None":
        """Confirm a VLM detection with a majority vote, rotating if needed.

        After the robot stops, pre-aligns toward the initial detection bbox
        (if provided) so the object is roughly centred before confirmation
        checks begin.  Then runs VLM ``confirm_checks`` times and requires at
        least ``confirm_threshold`` positives.  The first heading uses a more
        lenient threshold (``confirm_threshold - 1``, min 1) because the object
        was already detected once there during exploration.  If the object is
        not confirmed at the current heading, rotates by ``confirm_rotate_deg``
        in place (via ``UnitreeSkillContainer.relative_move``, which uses the
        A* planner and therefore checks the costmap for collision safety) and
        tries again.  Continues rotating until a full 360 degrees has been
        covered.

        Returns the best confirmed bbox, or None if the object was not
        confirmed after a full rotation scan.
        """
        checks = self.config.confirm_checks
        threshold = self.config.confirm_threshold
        # First heading gets a lenient threshold since we just detected it there.
        first_threshold = max(1, threshold - 1)
        step_deg = self.config.confirm_rotate_deg
        check_delay = self.config.confirm_check_delay
        total_rotated = 0.0

        # Pre-align toward the detection so we start confirmation facing the
        # object.  This is necessary when the detection bbox is near the edge
        # of the frame (common when the object enters the FOV while the robot
        # is moving).  Camera HFOV ≈ 38° (fx=460, width=320 from intrinsics).
        if initial_bbox is not None and self._latest_image is not None:
            img_w = self._latest_image.data.shape[1]
            cx = (initial_bbox[0] + initial_bbox[2]) / 2.0
            if initial_bbox[2] > img_w:  # bbox coords in 0-1000 scale
                cx = cx / 1000.0 * img_w
            # Fraction offset from image centre: [-1=far left, +1=far right]
            offset_ratio = (cx - img_w / 2.0) / (img_w / 2.0)
            # Camera horizontal half-FOV from intrinsics (fx≈460, width=320)
            hfov_half_deg = math.degrees(math.atan2(img_w / 2.0, 460.0))
            align_deg = round(offset_ratio * hfov_half_deg, 1)
            logger.info(
                "[VLN] Pre-aligning %.1f deg to centre detection (cx=%.0f/%d, offset=%.2f)",
                align_deg, cx, img_w, offset_ratio,
            )
            if abs(align_deg) > 3.0:
                try:
                    move_rpc = self.get_rpc_calls("UnitreeSkillContainer.relative_move")
                    move_rpc(0.0, 0.0, align_deg)
                    time.sleep(check_delay)
                except Exception:
                    logger.warning("[VLN] relative_move unavailable during pre-alignment")

        while total_rotated < 360.0:
            if self._search_stop.is_set():
                return None

            # Log current heading for diagnostics
            heading_deg: float | str = "unknown"
            if self._latest_odom is not None:
                heading_deg = round(
                    math.degrees(self._latest_odom.orientation.to_euler().z), 1
                )
            logger.info(
                "[VLN] Confirm '%s': heading %.1f deg rotated, yaw=%s, threshold=%d/%d",
                obj, total_rotated, heading_deg,
                first_threshold if total_rotated == 0.0 else threshold, checks,
            )

            # Majority vote: run VLM ``checks`` times at the current heading
            detections = 0
            best_bbox = None
            for check_idx in range(checks):
                if self._search_stop.is_set():
                    return None
                time.sleep(check_delay)  # wait for a fresh camera frame
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

            # Use lenient threshold for the initial (heading-0) check
            effective_threshold = first_threshold if total_rotated == 0.0 else threshold
            if detections >= effective_threshold:
                logger.info(
                    "[VLN] Object '%s' confirmed (%d/%d checks, threshold=%d) "
                    "after %.0f deg rotation",
                    obj, detections, checks, effective_threshold, total_rotated,
                )
                return best_bbox

            logger.info(
                "[VLN] Object '%s' not confirmed (%d/%d checks), rotating %.0f deg",
                obj, detections, checks, step_deg,
            )

            # Rotate in place — collision-safe via A* planner
            try:
                move_rpc = self.get_rpc_calls("UnitreeSkillContainer.relative_move")
                move_rpc(0.0, 0.0, step_deg)
            except Exception:
                logger.warning("[VLN] relative_move unavailable during confirmation scan")
                return None

            total_rotated += step_deg
            time.sleep(check_delay)  # let robot settle after rotation

        logger.info("[VLN] Object '%s' not confirmed after full 360 deg scan", obj)
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

                    if self._check_room_match(room):
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
        logger.info(f"[VLN] Starting exploration with active VLM search for '{obj}'")
        self._start_exploration()

        search_start = time.time()
        while time.time() - search_start < self.config.search_timeout:
            if self._search_stop.is_set():
                self._stop_exploration()
                return "Search cancelled."

            bbox, capture_odom = self._check_object_in_view(obj)
            if bbox:
                logger.info(
                    "[VLN] Found '%s' during exploration! (mode=%s)",
                    obj, self.config.exploration_mode,
                )

                # === CRITICAL: stop the robot IMMEDIATELY ===
                if self.config.exploration_mode == "navdp":
                    self._stop_exploration()
                else:
                    try:
                        cancel_rpc = self.get_rpc_calls("ReplanningAStarPlanner.cancel_goal")
                        cancel_rpc()
                    except Exception:
                        pass

                    self._stop_exploration()

                    try:
                        cancel_rpc = self.get_rpc_calls("ReplanningAStarPlanner.cancel_goal")
                        cancel_rpc()
                    except Exception:
                        pass

                # --- Overrun detection: navigate back if needed ---
                overrun = self._compute_overrun(capture_odom)
                if overrun > self.config.max_overrun_m:
                    logger.info(
                        "[VLN] Overrun %.1fm (> %.1fm) — navigating back to capture pose",
                        overrun, self.config.max_overrun_m,
                    )
                    self._navigate_to_capture_pose(capture_odom)

                # Confirmation phase: stop, majority-vote VLM, rotate+recheck
                # up to a full 360 degrees before giving up on this detection.
                # 1.5s settle ensures MuJoCo momentum has dissipated and the
                # camera has a stable, non-blurred frame before the first check.
                time.sleep(1.5)  # let robot settle after stop
                # Pass initial_bbox so _confirm_detection can pre-align the
                # robot toward the detection before majority-vote checks begin.
                bbox = self._confirm_detection(obj, initial_bbox=bbox)
                if bbox:
                    result = self._approach_object(bbox, obj)
                    return f"Found '{obj}' during exploration. {result}"
                else:
                    logger.info("[VLN] Detection not confirmed after 360 scan, resuming exploration")
                    self._start_exploration()
                    continue

            time.sleep(self.config.vlm_check_interval)

        self._stop_exploration()
        return f"Could not find '{obj}' within the search timeout ({self.config.search_timeout}s)."

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

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
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
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

    similarity_threshold: float = 0.35
    """Minimum CLIP similarity for semantic map queries."""

    semantic_near_pose_threshold: float = 0.5
    """Reject semantic-map matches whose stored pose is within this many metres
    of the robot's current odometry position.  Prevents trivial self-matches
    (where the best CLIP hit happens to be a frame captured at the robot's
    current location) from producing zero-distance A* goals that complete
    instantly and then falsely report failure."""

    exploration_mode: str = "astar"
    """Exploration backend: "astar" (WavefrontFrontier + A*), "navdp" (diffusion-policy nogoal),
    or "hybrid" (NavDP local obstacle avoidance + A* global path guidance)."""

    frontier_vlm_enabled: bool = False
    """In hybrid mode: use VLMFrontierJudge to rank frontier candidates before selecting goal."""

    frontier_vlm_interval: int = 5
    """In hybrid mode: re-run VLM frontier ranking every N frontier selections."""

    frontier_replan_distance_m: float = 3.0
    """In hybrid mode: recompute A* guidance path after the robot travels this many metres."""

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

    room_layout: str = ""
    """Path to a room layout YAML file (e.g. configs/sim/room_layout_house.yaml).
    When non-empty and the file exists, room centroids are seeded into SpatialMemory
    at startup via tag_location so VLN can navigate to named rooms immediately
    without requiring prior exploration.  Set to empty string to disable (default)."""

    # ------------------------------------------------------------------
    # Room-exit configuration
    # ------------------------------------------------------------------

    room_exit_enabled: bool = False
    """Enable memory-augmented room-exit mode when the robot saturates a room.
    When True, instead of returning failure on saturation, the robot attempts
    to backtrack along the entry trail and re-enter the search loop in an
    adjacent room.  Set to True in vln_config.yaml under room_exit.enabled."""

    room_exit_max_depth: int = 2
    """Maximum recursive room-exit attempts before giving up."""

    room_exit_vlm_judge_enabled: bool = True
    """Use Qwen3-VL-8B to evaluate frontier candidates during room-exit."""

    room_exit_vlm_judge_top_k: int = 5
    """Number of top geometric frontier candidates to show the VLM judge."""

    room_exit_vlm_judge_weight: float = 0.4
    """Blend weight for VLM confidence score (1 - this = geometric weight)."""

    room_exit_vlm_judge_timeout_s: float = 8.0
    """Maximum seconds to wait for VLM frontier judge response."""

    room_exit_trail_sample_spacing_m: float = 1.0
    """Minimum spacing (metres) between thinned trail waypoints."""

    room_exit_backtrack_novelty_threshold: float = 0.6
    """Stop backtracking when memory novelty score exceeds this."""

    room_exit_memory_query_radius_m: float = 1.5
    """SpatialMemory query radius when scoring trail waypoints."""

    room_exit_searched_area_radius_m: float = 2.0
    """Exclusion radius (metres) around tagged 'searched' locations."""

    room_exit_nav_timeout_s: float = 45.0
    """Timeout for navigating to the backtrack waypoint."""

    # ------------------------------------------------------------------
    # Multi-room discovery configuration
    # ------------------------------------------------------------------

    multi_room_enabled: bool = False
    """Enable two-phase multi-room search: first discover the correct room type,
    then search for the object within that room.  Set True for house/multi-room
    scenes."""

    multi_room_check_interval_m: float = 2.5
    """Minimum robot movement (metres) before re-identifying the current room type."""

    multi_room_max_rooms: int = 6
    """Give up room discovery after searching this many distinct rooms."""

    multi_room_discovery_timeout_s: float = 300.0
    """Hard time limit for the room discovery loop (seconds)."""

    multi_room_object_room_map: dict[str, list[str]] = {}  # type: ignore[assignment]
    """Hardcoded fallback: maps lower-case object keyword to a list of likely room types."""

    # ------------------------------------------------------------------
    # Memory-guided search configuration
    # ------------------------------------------------------------------

    use_memory: bool = True
    """Query SpatialMemory during the active search loop.
    When True, periodically re-queries CLIP semantic memory for the target object
    and navigates toward any match above similarity_threshold instead of continuing
    blind frontier exploration.  Set False for pure online exploration."""

    memory_query_interval: int = 5
    """Query memory every N VLM check cycles when use_memory=True.
    Lower values make memory queries more frequent; higher values rely more on
    frontier exploration between queries."""


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
        "SpatialMemory.query_by_location",
        "SpatialMemory.tag_location",
        "SpatialMemory.query_tagged_location",
        # Room-exit mode: corridor-aware frontier ranking + costmap for VLM rendering
        "WavefrontFrontierExplorer.get_frontiers_room_exit",
        "WavefrontFrontierExplorer.get_latest_costmap",
        # NavDP (optional — used for imagegoal approach when available)
        "NavDPNavigator.set_language_goal",
        "NavDPNavigator.set_reference_image",
        "NavDPNavigator.get_navdp_state",
        "NavDPNavigator.pause_motion",
        "NavDPNavigator.resume_motion",
        "NavDPNavigator.cancel_goal",
        "NavDPNavigator.get_exploration_stats",
        "NavDPNavigator.get_explore_trail",
        "NavDPNavigator.clear_exploration_trail",
        # In-place rotation during detection confirmation (uses A* planner, collision-safe)
        "UnitreeSkillContainer.relative_move",
        # NavDP memory — object instance recording (optional)
        "NavDPMemory.record_object_instance",
        "NavDPMemory.query_object_3d",
        # Hybrid exploration mode: A* path guidance for NavDP
        "ReplanningAStarPlanner.compute_path_to_goal",
        "NavDPNavigator.set_path_guidance",
        "NavDPNavigator.clear_path_guidance",
        "NavDPNavigator.get_escape_attempt_count",
        "WavefrontFrontierExplorer.get_best_frontier_goal",
        "WavefrontFrontierExplorer.mark_explored_goal",
    ]

    color_image: In[Image]
    depth_image: In[Image]
    odom: In[PoseStamped]
    realsense_camera_info: In[CameraInfo]

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

        # Multi-room room-identification cache
        # Stores (room_type_string, (x, y)) — invalidated when robot moves
        # more than multi_room_check_interval_m from the cached check position.
        self._room_id_cache: tuple[str, tuple[float, float]] | None = None

        # Memory-guided search state
        self._memory_vlm_cycle: int = 0
        """VLM cycle counter for memory_query_interval gating."""

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
        self._disposables.add(Disposable(self.realsense_camera_info.subscribe(self._on_camera_info)))

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

        if self.config.room_layout:
            seed_thread = threading.Thread(
                target=self._seed_rooms_delayed,
                args=(self.config.room_layout,),
                daemon=True,
                name="vln_room_seed",
            )
            seed_thread.start()

    @rpc
    def stop(self) -> None:
        self._cancel_search()
        super().stop()

    def _seed_rooms_delayed(self, path: str, delay: float = 5.0) -> None:
        """Background thread: wait for SpatialMemory to come up, then seed rooms."""
        time.sleep(delay)
        self._seed_rooms_from_layout(path)

    def _seed_rooms_from_layout(self, path: str) -> None:
        """Seed SpatialMemory with room centroids from a room layout YAML file.

        Reads rooms: {name: {x, y}} entries from the YAML and calls
        SpatialMemory.tag_location for each one so query_tagged_location
        can find rooms by name for immediate navigation without prior exploration.
        """
        import yaml

        layout_path = Path(path)
        if not layout_path.exists():
            logger.warning("[VLN] room_layout file not found: %s", path)
            return
        with open(layout_path) as f:
            layout = yaml.safe_load(f) or {}
        rooms = layout.get("rooms", {})
        if not rooms:
            logger.info("[VLN] room_layout has no rooms section: %s", path)
            return
        try:
            tag_rpc = self.get_rpc_calls("SpatialMemory.tag_location")
        except Exception as exc:
            logger.warning("[VLN] cannot seed rooms — SpatialMemory.tag_location unavailable: %s", exc)
            return
        from dimos.types.robot_location import RobotLocation

        seeded = 0
        for name, data in rooms.items():
            try:
                x = float(data.get("x", 0)) if isinstance(data, dict) else 0.0
                y = float(data.get("y", 0)) if isinstance(data, dict) else 0.0
                loc = RobotLocation(
                    name=name,
                    position=(x, y, 0.0),
                    rotation=(0.0, 0.0, 0.0),
                )
                tag_rpc(loc)
                seeded += 1
            except Exception as exc:
                logger.warning("[VLN] failed to seed room '%s': %s", name, exc)
        logger.info("[VLN] seeded %d rooms from %s into SpatialMemory", seeded, path)

    def _on_image(self, image: Image) -> None:
        self._latest_image = image

    def _on_depth(self, image: Image) -> None:
        self._latest_depth = image

    def _on_odom(self, odom: PoseStamped) -> None:
        self._latest_odom = odom

    def _on_camera_info(self, info: CameraInfo) -> None:
        """Reinitialize ObjectLocalizer and VLM auto_resize from live CameraInfo."""
        import numpy as np
        from dimos.navigation.visual.object_localizer import ObjectLocalizer

        K = np.array(info.K, dtype=np.float64).reshape(3, 3)
        self._localizer = ObjectLocalizer(
            intrinsic=K,
            cam_x=self.config.cam_x,
            cam_y=self.config.cam_y,
            cam_z=self.config.cam_z,
            cam_pitch=self.config.cam_pitch,
            depth_scale=self.config.depth_scale,
        )
        # Update navdp imagegoal size from live camera dimensions
        self.config.navdp_imagegoal_width = info.width
        self.config.navdp_imagegoal_height = info.height
        # Update VLM auto_resize so queries use the actual camera resolution
        if hasattr(self._vl_model, "config"):
            self._vl_model.config.auto_resize = (info.width, info.height)
        # logger.info(
        #     "[VLNSkill] CameraInfo updated: %dx%d fx=%.1f fy=%.1f",
        #     info.width, info.height, K[0, 0], K[1, 1],
        # )

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

        # NavDP and hybrid exploration must stay active across VLM checks so
        # stuck/escape history can accumulate; only pause motion.
        # Hybrid mode: the frontier loop thread must not be killed here, or
        # the A* guidance path is lost and escape state is reset every VLM cycle.
        # A* exploration still needs to be stopped explicitly (no pause primitive).
        if self.config.exploration_mode == "astar":
            try:
                self._stop_exploration()
            except Exception:
                pass

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
        # A* has no pause primitive so it must be fully restarted.
        # NavDP and hybrid modes only need motion resumed — the exploration
        # thread and escape state must not be reset here.
        if self.config.exploration_mode == "astar":
            self._start_exploration()

    def _run_vlm_check_while_paused(self, check_fn: Callable[[], T]) -> T:
        """Pause robot motion, run blocking VLM check, then return its result."""
        self._stop_motion_for_vlm_check()
        # Give the robot and camera a brief settle window.
        time.sleep(0.15)
        return check_fn()

    # ------------------------------------------------------------------
    # Multi-room helpers
    # ------------------------------------------------------------------

    _ROOM_TYPES = (
        "living room", "kitchen", "bedroom", "study room", "bathroom",
        "hallway", "corridor", "dining room", "garage", "office", "unknown",
    )

    def _identify_current_room(self) -> str:
        """Ask the VLM what type of room is currently visible.

        Result is cached by position: if the robot has not moved more than
        ``multi_room_check_interval_m`` since the last check, the cached
        value is returned immediately without a VLM call.

        Returns:
            A lower-case room type string (e.g. "study room", "kitchen",
            "living room", "unknown").
        """
        # --- position-based cache check ---
        current_pos = self._get_current_pose_xy()
        if current_pos and self._room_id_cache is not None:
            cached_room, cached_pos = self._room_id_cache
            dist = math.sqrt(
                (current_pos[0] - cached_pos[0]) ** 2
                + (current_pos[1] - cached_pos[1]) ** 2
            )
            if dist < self.config.multi_room_check_interval_m:
                return cached_room

        def _query() -> str:
            if self._latest_image is None:
                return "unknown"
            room_list = ", ".join(self._ROOM_TYPES)
            prompt = (
                "Look at this image from a robot camera mounted low inside a building.\n"
                f"What type of room is this? Choose the SINGLE best match from: {room_list}.\n"
                "Reply with ONLY the room type (e.g. 'study room') and nothing else."
            )
            try:
                response = self._vl_model.query(self._latest_image, prompt)
                response_lower = response.strip().lower()
                # Match against known room types (longest match first to prefer
                # "study room" over "room").
                for rt in sorted(self._ROOM_TYPES, key=len, reverse=True):
                    if rt in response_lower:
                        return rt
                # If no canonical match, return the first word(s) as-is (trimmed)
                return response_lower.split("\n")[0].strip()[:50]
            except Exception as exc:
                logger.warning("[VLN-MultiRoom] _identify_current_room VLM failed: %s", exc)
                return "unknown"

        room_type = self._run_vlm_check_while_paused(_query)
        # Update cache
        if current_pos:
            self._room_id_cache = (room_type, current_pos)
        logger.info(
            "[VLN-MultiRoom] Room identified as '%s' at pos=%s",
            room_type,
            f"({current_pos[0]:.2f}, {current_pos[1]:.2f})" if current_pos else "unknown",
        )
        return room_type

    def _infer_target_room_type(self, obj: str) -> list[str]:
        """Infer which room types are most likely to contain *obj*.

        First checks the hardcoded ``multi_room_object_room_map`` config dict.
        If no match is found, calls the VLM as a text-only reasoning step.

        Args:
            obj: Object name or description (e.g. "book case").

        Returns:
            Ordered list of lower-case room type strings (most likely first).
            Falls back to ``["unknown"]`` if all methods fail.
        """
        obj_lower = obj.lower()

        # --- Hardcoded map lookup (exact substring match) ---
        for keyword, room_types in self.config.multi_room_object_room_map.items():
            if keyword.lower() in obj_lower or obj_lower in keyword.lower():
                logger.info(
                    "[VLN-MultiRoom] Object '%s' matched keyword '%s' → rooms %s",
                    obj, keyword, room_types,
                )
                return [r.lower() for r in room_types]

        # --- VLM text-only reasoning ---
        room_list = ", ".join(self._ROOM_TYPES[:-1])  # exclude "unknown"
        prompt = (
            f"A mobile robot needs to find: \"{obj}\".\n"
            f"In a typical house or office, which room type(s) would most likely contain this item?\n"
            f"Available room types: {room_list}.\n"
            "List the 1-3 most likely room types, separated by commas, most likely first.\n"
            "Reply with ONLY the comma-separated list (e.g. 'study room, office')."
        )
        try:
            # Use text-only endpoint when available (Qwen3LocalVlModel), else
            # fall back to asking the VLM with a placeholder image.
            query_text_fn = getattr(self._vl_model, "query_text_only", None)
            if query_text_fn is not None:
                response = query_text_fn(prompt)
            elif self._latest_image is not None:
                response = self._vl_model.query(self._latest_image, prompt)
            else:
                raise RuntimeError("no text-only method and no image available")
            if not response:
                raise ValueError("empty response")
            # Parse comma-separated types, normalise
            raw_types = [t.strip().lower() for t in response.split(",")]
            # Filter to known room types
            matched: list[str] = []
            for raw in raw_types:
                for rt in self._ROOM_TYPES:
                    if rt in raw and rt not in matched:
                        matched.append(rt)
                        break
            if matched:
                logger.info(
                    "[VLN-MultiRoom] VLM inferred rooms for '%s': %s", obj, matched
                )
                return matched
        except Exception as exc:
            logger.warning(
                "[VLN-MultiRoom] _infer_target_room_type VLM failed: %s", exc
            )

        logger.info(
            "[VLN-MultiRoom] Could not infer room type for '%s', will search all rooms", obj
        )
        return ["unknown"]

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

    def _perform_observation_sweep(self) -> None:
        """Rotate 360° in 4×90° steps to collect scene observations before querying semantic memory.

        Each step: rotate 90° in-place, then pause 0.5 s for the camera /
        SpatialMemory to capture the new view.
        """
        logger.info("[VLN] observation sweep: starting 4×90° rotation")
        try:
            move_rpc = self.get_rpc_calls("UnitreeSkillContainer.relative_move")
        except Exception as e:
            logger.warning("[VLN] observation sweep: could not get relative_move RPC: %s", e)
            return
        for step in range(1, 5):
            logger.info("[VLN] observation sweep step %d/4 — rotating 90°", step)
            try:
                move_rpc(0.0, 0.0, 90.0)
                time.sleep(0.5)
            except Exception as e:
                logger.warning("[VLN] observation sweep step %d failed: %s", step, e)
        logger.info("[VLN] observation sweep: complete")

    # ------------------------------------------------------------------
    # Navigation helpers
    # ------------------------------------------------------------------

    def _navigate_to_semantic(self, query: str, do_sweep: bool = False) -> bool:
        """Try to navigate to a location found in the semantic map.

        Attempt 1: CLIP image-based semantic memory (populated during exploration).
        Attempt 2: Pre-seeded named locations via query_tagged_location (e.g. from
          room_layout_house.yaml seeded at startup). This allows immediate navigation
          to known rooms without requiring prior exploration.

        Args:
            query: text label to search for (room name, object description, …).
            do_sweep: if True, perform a 360° observation sweep first so the
                current scene has a fair chance to appear in semantic memory
                before committing to any stored match.
        """
        if do_sweep:
            self._perform_observation_sweep()

        try:
            set_goal_rpc = self.get_rpc_calls("ReplanningAStarPlanner.set_goal")
        except Exception:
            return False

        # Helper: return True if a candidate (x, y) is too close to the
        # robot's current position to be a useful navigation goal.
        def _is_near_current_pose(gx: float, gy: float) -> bool:
            if self._latest_odom is None:
                return False
            dx = gx - self._latest_odom.position.x
            dy = gy - self._latest_odom.position.y
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < self.config.semantic_near_pose_threshold:
                logger.info(
                    "[VLN] Semantic match at (%.2f, %.2f) is %.2fm from current pose"
                    " (threshold %.2fm) — skipping (self-match).",
                    gx, gy, dist, self.config.semantic_near_pose_threshold,
                )
                return True
            return False

        # Attempt 1: CLIP image-based semantic memory
        try:
            query_rpc = self.get_rpc_calls("SpatialMemory.query_by_text")
            results = query_rpc(query)
            if results:
                best = results[0]
                similarity = 1.0 - (best.get("distance") or 1)
                logger.info(
                    "[VLN] semantic CLIP query '%s': %d results, "
                    "best_similarity=%.4f threshold=%.2f → %s",
                    query, len(results), similarity,
                    self.config.similarity_threshold,
                    "ACCEPTED" if similarity >= self.config.similarity_threshold else "REJECTED (too low)",
                )
                if similarity >= self.config.similarity_threshold:
                    metadata = best.get("metadata")
                    if metadata:
                        first = metadata[0]
                        gx = first.get("pos_x", 0)
                        gy = first.get("pos_y", 0)
                        if not _is_near_current_pose(gx, gy):
                            pose = PoseStamped(
                                position=make_vector3(gx, gy, 0),
                                orientation=Quaternion.from_euler(
                                    make_vector3(0, 0, first.get("rot_z", 0))
                                ),
                                frame_id="map",
                            )
                            if set_goal_rpc(pose):
                                logger.info(
                                    "[VLN] CLIP match accepted: goal=(%0.2f, %.2f) "
                                    "similarity=%.4f — setting A* goal",
                                    gx, gy, similarity,
                                )
                                return True
                            else:
                                logger.warning(
                                    "[VLN] CLIP match at (%.2f, %.2f) rejected by A* planner "
                                    "(set_goal returned False)",
                                    gx, gy,
                                )
            else:
                logger.info("[VLN] semantic CLIP query '%s': no results returned", query)
        except Exception as e:
            logger.warning("[VLN] semantic CLIP query '%s' failed: %s", query, e)

        # Attempt 2: pre-seeded named locations (tag_location / room_layout seeding)
        try:
            tagged_rpc = self.get_rpc_calls("SpatialMemory.query_tagged_location")
            location = tagged_rpc(query)
            if location is not None:
                gx = float(location.position[0])
                gy = float(location.position[1])
                logger.info(
                    "[VLN] tagged location for '%s': name='%s' pos=(%.2f, %.2f)",
                    query, location.name, gx, gy,
                )
                if not _is_near_current_pose(gx, gy):
                    pose = PoseStamped(
                        position=make_vector3(gx, gy, 0),
                        orientation=Quaternion.from_euler(
                            make_vector3(0, 0, float(location.rotation[2]))
                        ),
                        frame_id="map",
                    )
                    result = set_goal_rpc(pose)
                    logger.info(
                        "[VLN] tagged location goal set result=%s for '%s' at (%.2f, %.2f)",
                        result, query, gx, gy,
                    )
                    return result
            else:
                logger.info("[VLN] no tagged location found for '%s'", query)
        except Exception as e:
            logger.warning("[VLN] tagged location query '%s' failed: %s", query, e)

        logger.info("[VLN] _navigate_to_semantic('%s'): all attempts failed → returning False", query)
        return False

    def _wait_for_navigation(self, timeout: float = 30.0) -> bool:
        """Block until navigation finishes or timeout. Returns True if goal reached.

        A short initial sleep of 0.05 s is inserted before the first poll so
        the planner has time to transition *away* from its pre-navigation IDLE
        state.  Without this, a trivial (zero-distance) goal completes in < 1 ms
        and the planner is already back to IDLE before the first 0.5 s poll fires;
        ``is_goal_reached()`` then returns its stale pre-reset value (False),
        causing a false-negative "could not reach" report.
        """
        try:
            get_state_rpc, is_reached_rpc = self.get_rpc_calls(
                "ReplanningAStarPlanner.get_state", "ReplanningAStarPlanner.is_goal_reached"
            )
        except Exception:
            return False

        # Give the planner a moment to leave IDLE and enter path_following
        # (or arrive immediately for a near-zero-distance goal).
        time.sleep(0.05)

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
        if self.config.exploration_mode == "hybrid":
            return self._start_exploration_hybrid()
        return self._start_exploration_astar()

    def _stop_exploration(self) -> bool:
        if self.config.exploration_mode == "navdp":
            return self._stop_exploration_navdp()
        if self.config.exploration_mode == "hybrid":
            return self._stop_exploration_hybrid()
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

    # -- Hybrid exploration (NavDP local + A* global path guidance) --

    def _start_exploration_hybrid(self) -> bool:
        """Start hybrid exploration: NavDP handles local obstacle avoidance,
        A* provides global path guidance via set_path_guidance.

        Starts NavDP in SEEK state, then launches _hybrid_frontier_loop in a
        background thread to periodically select a frontier goal and pipe the
        A* path as directional guidance to NavDP's trajectory selector.
        """
        try:
            set_lang_rpc = self.get_rpc_calls("NavDPNavigator.set_language_goal")
            set_lang_rpc("explore the environment")
            logger.info("[VLN] Hybrid exploration started (NavDP+A*)")
        except Exception:
            logger.warning("[VLN] NavDPNavigator not connected, falling back to A*")
            return self._start_exploration_astar()

        self._hybrid_stop_event = threading.Event()
        self._hybrid_thread = threading.Thread(
            target=self._hybrid_frontier_loop, daemon=True, name="hybrid-frontier-loop"
        )
        self._hybrid_thread.start()
        return True

    def _stop_exploration_hybrid(self) -> bool:
        """Stop hybrid exploration: cancel NavDP and shut down the frontier loop."""
        # Signal the background loop to stop
        stop_event = getattr(self, "_hybrid_stop_event", None)
        if stop_event is not None:
            stop_event.set()

        thread = getattr(self, "_hybrid_thread", None)
        if thread is not None and thread.is_alive():
            thread.join(timeout=3.0)

        # Clear guidance path so NavDP falls back to frontier direction
        try:
            clear_rpc = self.get_rpc_calls("NavDPNavigator.clear_path_guidance")
            clear_rpc()
        except Exception:
            pass

        # Cancel NavDP motion
        try:
            cancel_rpc = self.get_rpc_calls("NavDPNavigator.cancel_goal")
            cancel_rpc()
        except Exception:
            pass

        logger.info("[VLN] Hybrid exploration stopped")
        return True

    def _hybrid_frontier_loop(self) -> None:
        """Background thread: select frontier → compute A* path → push to NavDP.

        Runs at most once every ``frontier_replan_distance_m`` metres of travel.
        If VLM frontier ranking is enabled, runs VLMFrontierJudge asynchronously;
        the result is applied on the next selection cycle so it never blocks motion.
        """
        stop_event: threading.Event = self._hybrid_stop_event
        replan_dist = self.config.frontier_replan_distance_m
        vlm_enabled = self.config.frontier_vlm_enabled
        vlm_interval = max(1, self.config.frontier_vlm_interval)

        last_replan_pos: tuple[float, float] | None = None
        frontier_sel_count = 0
        pending_vlm_scores: dict[int, float] | None = None  # from last async VLM call
        pending_frontier_candidates: list[Any] = []
        current_frontier_goal: Any = None   # last frontier goal pushed to NavDP
        last_escape_count: int = 0          # escape_attempt_count at last replan

        def _run_vlm_async(candidates: list[Any]) -> None:
            """Fire-and-forget VLM frontier ranking — writes result into closure."""
            nonlocal pending_vlm_scores, pending_frontier_candidates
            try:
                from dimos.navigation.vlm_frontier_judge import VLMFrontierJudge, blend_scores  # noqa: F401

                judge = VLMFrontierJudge(
                    vlm_base_url=self.config.vlm_base_url,
                    timeout_s=self.config.room_exit_vlm_judge_timeout_s,
                    vlm_judge_top_k=len(candidates),
                )
                # Build geo scores (uniform, since we just want VLM ranking here)
                geo_scores = {i: 1.0 for i in range(len(candidates))}
                judge_result = judge.rank_frontiers(
                    frontiers=candidates,
                    costmap=None,
                    trail=None,
                )
                if not judge_result.fell_back_to_uniform:
                    pending_frontier_candidates = candidates
                    pending_vlm_scores = judge_result.confidence_map()
            except Exception as exc:
                logger.debug("[VLN-Hybrid] VLM frontier judge failed: %s", exc)

        while not stop_event.is_set():
            stop_event.wait(timeout=0.5)
            if stop_event.is_set():
                break

            # Get current odometry
            odom = self._latest_odom
            if odom is None:
                continue

            ox, oy = odom.position.x, odom.position.y

            # Check whether NavDP has become newly stuck since the last replan.
            # If escape_attempt_count has increased, the current frontier goal
            # is unreachable from this position — blacklist it and replan now.
            escape_count: int = 0
            try:
                esc_rpc = self.get_rpc_calls("NavDPNavigator.get_escape_attempt_count")
                escape_count = esc_rpc() or 0
            except Exception:
                pass

            newly_stuck = escape_count > last_escape_count
            if newly_stuck and current_frontier_goal is not None:
                logger.info(
                    "[VLN-Hybrid] NavDP stuck (escape_count=%d→%d) — blacklisting "
                    "frontier (%.2f, %.2f) and forcing replan",
                    last_escape_count, escape_count,
                    current_frontier_goal.x, current_frontier_goal.y,
                )
                try:
                    mark_rpc = self.get_rpc_calls("WavefrontFrontierExplorer.mark_explored_goal")
                    mark_rpc(current_frontier_goal)
                except Exception:
                    pass
                # Clear guidance so NavDP doesn't keep chasing the dead-end path
                try:
                    clear_rpc = self.get_rpc_calls("NavDPNavigator.clear_path_guidance")
                    clear_rpc()
                except Exception:
                    pass
                current_frontier_goal = None
                last_replan_pos = None  # force immediate replan below

            last_escape_count = escape_count

            # Only replan after the robot has travelled far enough (or forced above)
            if last_replan_pos is not None:
                dist = math.hypot(ox - last_replan_pos[0], oy - last_replan_pos[1])
                if dist < replan_dist:
                    continue

            # Get best frontier goal from WavefrontFrontierExplorer
            goal_vec: Any = None
            try:
                frontier_rpc = self.get_rpc_calls("WavefrontFrontierExplorer.get_best_frontier_goal")
                goal_vec = frontier_rpc()
            except Exception as exc:
                logger.debug("[VLN-Hybrid] get_best_frontier_goal failed: %s", exc)
                continue

            if goal_vec is None:
                logger.info("[VLN-Hybrid] No frontier available — waiting")
                continue

            frontier_sel_count += 1

            # Optionally kick off async VLM ranking every N selections
            if vlm_enabled and frontier_sel_count % vlm_interval == 0:
                try:
                    frontiers_rpc = self.get_rpc_calls("WavefrontFrontierExplorer.get_frontiers_room_exit")
                    candidates_raw = frontiers_rpc(self.config.room_exit_vlm_judge_top_k)
                    if candidates_raw:
                        candidates = [f for f, _ in candidates_raw]
                        threading.Thread(
                            target=_run_vlm_async, args=(candidates,), daemon=True
                        ).start()
                except Exception:
                    pass

            # If pending VLM results match current candidates, use the best one
            if pending_vlm_scores and pending_frontier_candidates:
                best_idx = max(pending_vlm_scores, key=lambda i: pending_vlm_scores[i])  # type: ignore[arg-type]
                if best_idx < len(pending_frontier_candidates):
                    goal_vec = pending_frontier_candidates[best_idx]
                    pending_vlm_scores = None
                    pending_frontier_candidates = []

            # Compute A* path to the selected frontier goal
            path_obj: Any = None
            try:
                from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped as PS

                goal_pose = PS()
                goal_pose.position.x = float(goal_vec.x)
                goal_pose.position.y = float(goal_vec.y)
                goal_pose.position.z = 0.0
                goal_pose.orientation.w = 1.0
                goal_pose.frame_id = "world"

                compute_rpc = self.get_rpc_calls("ReplanningAStarPlanner.compute_path_to_goal")
                path_obj = compute_rpc(goal_pose)
            except Exception as exc:
                logger.debug("[VLN-Hybrid] compute_path_to_goal failed: %s", exc)

            if path_obj is None or not getattr(path_obj, "poses", None):
                logger.debug("[VLN-Hybrid] No A* path to frontier (%.2f, %.2f)", goal_vec.x, goal_vec.y)
                # Mark as explored so we don't keep picking an unreachable frontier
                try:
                    mark_rpc = self.get_rpc_calls("WavefrontFrontierExplorer.mark_explored_goal")
                    mark_rpc(goal_vec)
                except Exception:
                    pass
                continue

            # Push waypoints to NavDP as path guidance
            waypoints = [[p.position.x, p.position.y] for p in path_obj.poses]
            try:
                set_path_rpc = self.get_rpc_calls("NavDPNavigator.set_path_guidance")
                set_path_rpc(waypoints)
                logger.info(
                    "[VLN-Hybrid] New A* guidance path: %d waypoints → frontier (%.2f, %.2f)",
                    len(waypoints),
                    goal_vec.x,
                    goal_vec.y,
                )
                current_frontier_goal = goal_vec
            except Exception as exc:
                logger.debug("[VLN-Hybrid] set_path_guidance failed: %s", exc)

            last_replan_pos = (ox, oy)

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
                # goal_w = self.config.navdp_imagegoal_width
                # goal_h = self.config.navdp_imagegoal_height
                # ref_bgr = cv2.resize(ref_bgr, (goal_w, goal_h), interpolation=cv2.INTER_LINEAR)
                # logger.info(
                #     "[VLN] Resized imagegoal to %dx%d for NavDP", goal_w, goal_h
                # )
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

    # ------------------------------------------------------------------
    # Room-exit helpers
    # ------------------------------------------------------------------

    def _get_current_pose_xy(self) -> tuple[float, float] | None:
        """Return the current (x, y) robot pose, or None if odom unavailable."""
        odom = self._latest_odom
        if odom is None:
            return None
        return (float(odom.position.x), float(odom.position.y))

    def _navigate_to_waypoint_astar(self, x: float, y: float, timeout: float = 45.0) -> bool:
        """Navigate to a world-frame (x, y) waypoint via A* planner.

        Args:
            x: Target x position (metres, world frame).
            y: Target y position (metres, world frame).
            timeout: Maximum navigation time in seconds.

        Returns:
            True if the waypoint was reached.
        """
        try:
            set_goal_rpc = self.get_rpc_calls("ReplanningAStarPlanner.set_goal")
        except Exception:
            return False

        goal = PoseStamped(
            position=make_vector3(x, y, 0.0),
            orientation=Quaternion.from_euler(make_vector3(0.0, 0.0, 0.0)),
            frame_id="map",
        )
        try:
            set_goal_rpc(goal)
        except Exception as exc:
            logger.warning("[VLN-Exit] set_goal failed: %s", exc)
            return False

        return self._wait_for_navigation(timeout=timeout)

    def _tag_searched_area(self, x: float, y: float) -> None:
        """Tag the current area as searched in SpatialMemory.

        Uses a structured name ``searched:room:{timestamp}`` so that future
        calls to ``query_tagged_location("searched:room")`` can locate and
        avoid re-entered areas.
        """
        try:
            tag_rpc = self.get_rpc_calls("SpatialMemory.tag_location")
            from dimos.types.robot_location import RobotLocation
            from dimos.navigation.room_exit_planner import RoomExitPlanner

            name = RoomExitPlanner.make_searched_tag_name()
            tag_rpc(
                RobotLocation(
                    name=name,
                    position=(x, y, 0.0),
                    rotation=(0.0, 0.0, 0.0),
                )
            )
            logger.info("[VLN-Exit] Tagged searched area at (%.2f, %.2f) as '%s'", x, y, name)
        except Exception as exc:
            logger.warning("[VLN-Exit] Could not tag searched area: %s", exc)

    def _get_searched_locations(self) -> list[tuple[float, float]]:
        """Return all previously tagged 'searched' room positions.

        Returns:
            List of (x, y) world-frame positions already searched.
        """
        positions: list[tuple[float, float]] = []
        try:
            tagged_rpc = self.get_rpc_calls("SpatialMemory.query_tagged_location")
            loc = tagged_rpc("searched:room")
            if loc is not None:
                positions.append((float(loc.position[0]), float(loc.position[1])))
        except Exception:
            pass
        return positions

    def _room_discovery_loop(
        self,
        obj: str,
        candidate_room_types: list[str],
    ) -> str | None:
        """Explore until reaching a room whose type matches *candidate_room_types*.

        This is Phase 1B of the multi-room search.  The robot:

        1. Starts NavDP nogoal exploration.
        2. Every ``multi_room_check_interval_m`` metres, calls
           ``_identify_current_room()`` and tags the result in SpatialMemory.
        3. If the identified room type is in *candidate_room_types*, stops
           exploration and returns ``None`` (caller falls through to object
           search).
        4. If the room saturates (coverage threshold reached or stall timeout),
           calls ``_room_exit_sequence`` to move to an adjacent area and
           continues discovery there.
        5. Gives up after ``multi_room_max_rooms`` distinct rooms or
           ``multi_room_discovery_timeout_s`` seconds.

        Args:
            obj: Target object — used only if the object is serendipitously
                spotted during discovery (early return with result string).
            candidate_room_types: Lower-case room type strings that indicate
                the robot is in the right room (e.g. ``["study room", "office"]``).

        Returns:
            Non-None result string if the object was found during discovery
            (unlikely but possible), ``None`` if the correct room was reached
            or the search is cancelled.
        """
        if self._search_stop.is_set():
            return None

        logger.info(
            "[VLN-Discovery] Starting room discovery for '%s', target room types: %s",
            obj, candidate_room_types,
        )

        # When candidate_room_types is ["unknown"], we accept any room and
        # immediately fall through to object search.
        accept_any_room = candidate_room_types == ["unknown"]

        discovery_start = time.time()
        rooms_searched: list[str] = []
        last_pos = self._get_current_pose_xy()
        last_room_check_pos = last_pos

        # Tag for RoomExitPlanner anti-re-entry
        tag_rpc = None
        try:
            tag_rpc = self.get_rpc_calls("SpatialMemory.tag_location")
        except Exception:
            pass

        def _tag_room(room_type: str) -> None:
            if tag_rpc is None or room_type in ("unknown",):
                return
            try:
                from dimos.types.robot_location import RobotLocation
                pos = self._get_current_pose_xy()
                if pos:
                    tag_rpc(
                        RobotLocation(
                            name=f"room:{room_type}",
                            position=(pos[0], pos[1], 0.0),
                            rotation=(0.0, 0.0, 0.0),
                        )
                    )
            except Exception as exc:
                logger.debug("[VLN-Discovery] Could not tag room '%s': %s", room_type, exc)

        # Start exploration
        self._start_exploration()

        last_progress_time = discovery_start
        last_trail_distance = 0.0

        try:
            stats_rpc = self.get_rpc_calls("NavDPNavigator.get_exploration_stats")
            init_stats = stats_rpc()
            last_trail_distance = init_stats.get("trail_distance_m", 0.0)
        except Exception:
            stats_rpc = None

        while True:
            if self._search_stop.is_set():
                self._stop_exploration()
                return None

            now = time.time()
            elapsed = now - discovery_start

            # Hard timeout for discovery phase
            if elapsed > self.config.multi_room_discovery_timeout_s:
                logger.info(
                    "[VLN-Discovery] Hard timeout (%.0fs), stopping discovery", elapsed
                )
                break

            # If we've already saturated "too many" rooms, give up
            if len(rooms_searched) >= self.config.multi_room_max_rooms:
                logger.info(
                    "[VLN-Discovery] Searched %d distinct rooms without finding a match, giving up",
                    len(rooms_searched),
                )
                break

            # --- Stall / saturation detection (reuse existing stats signals) ---
            stalled_for = now - last_progress_time
            room_saturated = False

            if stats_rpc is not None:
                try:
                    stats = stats_rpc()
                    cur_trail_dist = stats.get("trail_distance_m", 0.0)
                    cur_observed = stats.get("observed_fraction", 0.0)

                    new_distance = cur_trail_dist - last_trail_distance
                    if new_distance >= self.config.search_progress_distance:
                        last_progress_time = now
                        last_trail_distance = cur_trail_dist

                    if (
                        stats.get("costmap_available", False)
                        and cur_observed >= self.config.search_observed_fraction
                    ):
                        room_saturated = True
                except Exception:
                    pass

            if stalled_for > self.config.search_stall_timeout:
                room_saturated = True

            # --- Check if the robot has moved enough to re-identify the room ---
            current_pos = self._get_current_pose_xy()
            if current_pos and last_room_check_pos:
                moved = math.sqrt(
                    (current_pos[0] - last_room_check_pos[0]) ** 2
                    + (current_pos[1] - last_room_check_pos[1]) ** 2
                )
            else:
                moved = self.config.multi_room_check_interval_m  # force a check at startup

            if moved >= self.config.multi_room_check_interval_m or (elapsed < 2.0 and not rooms_searched):
                # Identify current room
                room_type = self._identify_current_room()
                last_room_check_pos = current_pos

                if room_type not in rooms_searched:
                    rooms_searched.append(room_type)
                    _tag_room(room_type)
                    logger.info(
                        "[VLN-Discovery] New room identified: '%s' (searched so far: %s)",
                        room_type, rooms_searched,
                    )

                # If we accept any room OR this room matches our target types
                if accept_any_room or room_type in candidate_room_types:
                    logger.info(
                        "[VLN-Discovery] Target room type '%s' found! Stopping discovery.",
                        room_type,
                    )
                    self._stop_exploration()
                    return None  # fall through to object search

                # If the current room is wrong AND saturated, exit immediately
                if room_saturated:
                    logger.info(
                        "[VLN-Discovery] Room '%s' saturated (wrong type). Triggering room exit.",
                        room_type,
                    )
                    self._stop_exploration()
                    # Tag as searched and attempt room exit
                    if current_pos:
                        self._tag_searched_area(current_pos[0], current_pos[1])
                    exit_result = self._room_exit_sequence(obj, depth=0)
                    if exit_result is not None:
                        return exit_result
                    # After exit, continue the discovery loop from the new position
                    self._start_exploration()
                    last_progress_time = time.time()
                    last_trail_distance = 0.0
                    if stats_rpc is not None:
                        try:
                            init_stats = stats_rpc()
                            last_trail_distance = init_stats.get("trail_distance_m", 0.0)
                        except Exception:
                            pass
                    # Reset room check position so we re-identify immediately
                    last_room_check_pos = self._get_current_pose_xy()
                    continue

            elif room_saturated:
                # Room is saturated but we haven't moved enough to re-identify
                # — exit anyway and continue discovery
                logger.info("[VLN-Discovery] Room saturated before re-identification. Exiting.")
                self._stop_exploration()
                current_pos = self._get_current_pose_xy()
                if current_pos:
                    self._tag_searched_area(current_pos[0], current_pos[1])
                exit_result = self._room_exit_sequence(obj, depth=0)
                if exit_result is not None:
                    return exit_result
                self._start_exploration()
                last_progress_time = time.time()
                last_trail_distance = 0.0
                if stats_rpc is not None:
                    try:
                        init_stats = stats_rpc()
                        last_trail_distance = init_stats.get("trail_distance_m", 0.0)
                    except Exception:
                        pass
                last_room_check_pos = self._get_current_pose_xy()
                continue

            # --- Opportunistic object check during discovery ---
            # Don't do a full VLM object check here — it would slow down
            # discovery significantly.  We rely on the subsequent object search
            # phase for that.  But we DO do a quick check every N seconds as a
            # shortcut if the object happens to be right in front of us.

            time.sleep(self.config.vlm_check_interval)

        # Discovery loop ended without finding the target room
        self._stop_exploration()
        logger.info(
            "[VLN-Discovery] Room discovery ended without finding target room types %s. "
            "Searched rooms: %s",
            candidate_room_types, rooms_searched,
        )
        return None

    def _room_exit_sequence(
        self, obj: str, depth: int = 0
    ) -> str | None:
        """Attempt to exit the saturated room and continue the search elsewhere.

        Orchestrates the full room-exit harness:
        1. Tag the current area as 'searched' in SpatialMemory.
        2. Retrieve the NavDP exploration trail.
        3. Use RoomExitPlanner to find the highest-novelty backtrack waypoint.
        4. Navigate to that waypoint (A* planner).
        5. Get frontier candidates from WavefrontFrontierExplorer (room-exit mode).
        6. Optionally invoke VLMFrontierJudge to rank candidates semantically.
        7. Fuse geometric + VLM scores and navigate to the best frontier.
        8. Resume active search in the new area (recursive, depth-limited).

        Args:
            obj: The target object being searched for.
            depth: Current recursion depth (max ``room_exit_max_depth``).

        Returns:
            Non-None string result if the object was found during re-entry, or
            None to signal that exit failed / object not found (caller should
            fall back to returning failure).
        """
        if depth >= self.config.room_exit_max_depth:
            logger.info("[VLN-Exit] Max room-exit depth (%d) reached, giving up", depth)
            return None

        if self._search_stop.is_set():
            return None

        logger.info("[VLN-Exit] Starting room-exit sequence (depth=%d) for '%s'", depth, obj)

        # --- Step 1: Tag current area as searched ---
        pose = self._get_current_pose_xy()
        if pose:
            self._tag_searched_area(pose[0], pose[1])

        # --- Step 2: Get exploration trail ---
        raw_trail: list[tuple[float, float]] = []
        try:
            trail_rpc = self.get_rpc_calls("NavDPNavigator.get_explore_trail")
            raw_trail = trail_rpc() or []
            logger.info("[VLN-Exit] Retrieved trail with %d points", len(raw_trail))
        except Exception as exc:
            logger.warning("[VLN-Exit] Could not get explore trail: %s", exc)

        # --- Step 3: Find backtrack target ---
        from dimos.navigation.room_exit_planner import RoomExitConfig, RoomExitPlanner

        exit_config = RoomExitConfig(
            trail_sample_spacing_m=self.config.room_exit_trail_sample_spacing_m,
            backtrack_novelty_threshold=self.config.room_exit_backtrack_novelty_threshold,
            memory_query_radius_m=self.config.room_exit_memory_query_radius_m,
        )

        query_by_location_fn = None
        query_by_text_fn = None
        try:
            query_by_location_fn = self.get_rpc_calls("SpatialMemory.query_by_location")
        except Exception:
            pass
        try:
            query_by_text_fn = self.get_rpc_calls("SpatialMemory.query_by_text")
        except Exception:
            pass

        planner = RoomExitPlanner(
            config=exit_config,
            query_by_location_fn=query_by_location_fn,
            query_by_text_fn=query_by_text_fn,
        )

        backtrack_target = planner.find_backtrack_target(raw_trail)
        if backtrack_target is None:
            logger.info("[VLN-Exit] No backtrack target found (trail too short)")
            return None

        logger.info(
            "[VLN-Exit] Backtrack target: (%.2f, %.2f) novelty=%.2f",
            backtrack_target.x, backtrack_target.y, backtrack_target.novelty_score,
        )

        # --- Step 4: Navigate to backtrack waypoint ---
        if self._search_stop.is_set():
            return None

        reached_backtrack = self._navigate_to_waypoint_astar(
            backtrack_target.x, backtrack_target.y,
            timeout=self.config.room_exit_nav_timeout_s,
        )
        if not reached_backtrack:
            logger.info("[VLN-Exit] Could not reach backtrack waypoint, trying from current pos")

        # --- Step 5 & 6: Get frontier candidates + VLM judge ---
        current_odom = self._latest_odom
        if current_odom is None:
            logger.info("[VLN-Exit] No odometry available, cannot score frontiers")
            return None

        robot_pos_vec = current_odom.position

        frontier_candidates: list[Any] = []
        geo_scores: dict[int, float] = {}

        try:
            wavefront_rpc = self.get_rpc_calls("WavefrontFrontierExplorer.get_frontiers_room_exit")
            # get_frontiers_room_exit uses the module's internal costmap
            scored_frontiers = wavefront_rpc(self.config.room_exit_vlm_judge_top_k)
            if scored_frontiers:
                for i, (frontier, score) in enumerate(scored_frontiers):
                    frontier_candidates.append(frontier)
                    geo_scores[i] = score
                logger.info("[VLN-Exit] Got %d room-exit frontier candidates", len(frontier_candidates))
        except Exception as exc:
            logger.warning("[VLN-Exit] WavefrontFrontierExplorer.get_frontiers_room_exit unavailable: %s", exc)

        if not frontier_candidates:
            logger.info("[VLN-Exit] No frontiers available after backtrack — resuming standard search")
            self._start_exploration()
            return None

        # VLM judge: score frontiers semantically
        best_frontier_idx = 0  # default to highest geometric score
        if self.config.room_exit_vlm_judge_enabled and len(frontier_candidates) > 1:
            try:
                from dimos.navigation.vlm_frontier_judge import VLMFrontierJudge, blend_scores

                judge = VLMFrontierJudge(
                    vlm_base_url=self.config.vlm_base_url,
                    timeout_s=self.config.room_exit_vlm_judge_timeout_s,
                    vlm_judge_top_k=self.config.room_exit_vlm_judge_top_k,
                )

                # Fetch latest costmap from WavefrontFrontierExplorer for rendering
                latest_costmap = None
                try:
                    costmap_rpc = self.get_rpc_calls("WavefrontFrontierExplorer.get_latest_costmap")
                    latest_costmap = costmap_rpc()
                except Exception:
                    pass

                judge_result = judge.judge_frontiers(
                    grid=latest_costmap,
                    frontiers=frontier_candidates,
                    robot_pos=robot_pos_vec,
                    trail=raw_trail,
                )

                if not judge_result.fell_back_to_uniform:
                    fused = blend_scores(
                        geo_scores,
                        judge_result.confidence_map(),
                        geo_weight=1.0 - self.config.room_exit_vlm_judge_weight,
                        vlm_weight=self.config.room_exit_vlm_judge_weight,
                    )
                    best_frontier_idx = max(fused, key=fused.get)  # type: ignore[arg-type]
                    logger.info(
                        "[VLN-Exit] VLM judge selected frontier %d. "
                        "Reason: %s",
                        best_frontier_idx + 1,
                        judge_result.judgements[0].reason if judge_result.judgements else "n/a",
                    )
                else:
                    logger.info("[VLN-Exit] VLM judge fell back to uniform, using geometric ranking")
            except Exception as exc:
                logger.warning("[VLN-Exit] VLM judge failed: %s", exc)

        best_frontier = frontier_candidates[best_frontier_idx]
        logger.info(
            "[VLN-Exit] Navigating to best exit frontier: (%.2f, %.2f)",
            best_frontier.x, best_frontier.y,
        )

        # --- Step 7: Navigate to selected frontier ---
        if self._search_stop.is_set():
            return None

        self._navigate_to_waypoint_astar(
            best_frontier.x, best_frontier.y,
            timeout=self.config.room_exit_nav_timeout_s,
        )

        # --- Step 8: Resume active search in new area (recursive) ---
        logger.info("[VLN-Exit] Entered new area — resuming active search for '%s'", obj)
        return self._active_search_loop(obj, depth=depth + 1)

    def _active_search_loop(self, obj: str, depth: int = 0) -> str | None:
        """Run the active VLM search loop in the current area.

        Extracted from the inline loop in ``_vln_search`` so it can be called
        recursively by ``_room_exit_sequence``.

        Args:
            obj: Target object description.
            depth: Room-exit recursion depth (passed through to _room_exit_sequence).

        Returns:
            Non-None success string if the object is found, None otherwise.
        """
        if self._search_stop.is_set():
            return None

        self._start_exploration()

        search_start = time.time()
        last_progress_time = search_start
        last_trail_distance = 0.0
        last_observed_fraction = 0.0
        self._memory_vlm_cycle = 0

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

            # Hard timeout
            if elapsed > self.config.search_timeout:
                logger.info("[VLN-ActiveSearch] Hard timeout (%.0fs)", elapsed)
                break

            # Stall timeout
            if stalled_for > self.config.search_stall_timeout:
                logger.info("[VLN-ActiveSearch] Stall timeout (no new coverage for %.0fs)", stalled_for)
                if self.config.room_exit_enabled:
                    self._stop_exploration()
                    return self._room_exit_sequence(obj, depth=depth)
                break

            # Cancelled externally
            if self._search_stop.is_set():
                self._stop_exploration()
                return None

            # VLM check
            bbox, capture_odom = self._check_object_in_view(obj, resume_search=True)
            if bbox:
                logger.info("[VLN-ActiveSearch] Found '%s' during exploration!", obj)
                self._stop_exploration()
                time.sleep(1.5)
                bbox = self._confirm_detection(obj)
                if bbox:
                    result = self._approach_object(bbox, obj)
                    return f"Found '{obj}' during exploration. {result}"
                else:
                    logger.info("[VLN-ActiveSearch] Detection not confirmed, resuming")
                    self._start_exploration()
                    last_progress_time = time.time()
                    self._memory_vlm_cycle = 0
                    continue

            # Memory query: periodically re-query SpatialMemory for the target
            # object and navigate toward any match found above similarity_threshold.
            # Only runs when use_memory=True and every memory_query_interval VLM cycles.
            self._memory_vlm_cycle += 1
            if (
                self.config.use_memory
                and self._memory_vlm_cycle % self.config.memory_query_interval == 0
            ):
                logger.info(
                    "[VLN-ActiveSearch] Memory query for '%s' (cycle %d)",
                    obj, self._memory_vlm_cycle,
                )
                self._stop_exploration()
                found = self._navigate_to_semantic(obj)
                if found:
                    logger.info(
                        "[VLN-ActiveSearch] Memory match found — navigating to stored location"
                    )
                    last_progress_time = time.time()
                else:
                    logger.info(
                        "[VLN-ActiveSearch] No memory match above threshold — resuming exploration"
                    )
                    self._start_exploration()

            # Progress check
            if stats_rpc is not None:
                try:
                    stats = stats_rpc()
                    cur_trail_dist = stats.get("trail_distance_m", 0.0)
                    cur_observed = stats.get("observed_fraction", 0.0)

                    new_distance = cur_trail_dist - last_trail_distance
                    if new_distance >= self.config.search_progress_distance:
                        last_progress_time = time.time()
                        last_trail_distance = cur_trail_dist
                        last_observed_fraction = cur_observed

                    # Boundary fully observed
                    if (
                        stats.get("costmap_available", False)
                        and cur_observed >= self.config.search_observed_fraction
                    ):
                        logger.info(
                            "[VLN-ActiveSearch] Boundary fully observed: %.1f%% known. "
                            "Triggering room-exit.",
                            cur_observed * 100,
                        )
                        if self.config.room_exit_enabled:
                            self._stop_exploration()
                            return self._room_exit_sequence(obj, depth=depth)
                        break
                except Exception:
                    pass

            time.sleep(self.config.vlm_check_interval)

        self._stop_exploration()
        return None

    def _vln_search(self, room: str | None, obj: str | None) -> str:
        """Execute the full VLN pipeline. Runs in a worker thread."""
        # Phase 1 — Navigate to room (if specified)
        if room:
            logger.info(f"[VLN] Phase 1: navigating to room '{room}'")

            # First, try the semantic map — sweep 360° so the current
            # scene is captured before querying stored embeddings.
            if self._navigate_to_semantic(room, do_sweep=True):
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

        # Phase 2B — Multi-room discovery (only when room was NOT explicitly named)
        # If room=None and multi_room_mode is on, infer which room type is most
        # likely to contain the object, then explore until we reach that room.
        if not room and self.config.multi_room_enabled:
            logger.info("[VLN] Multi-room mode: inferring target room type for '%s'", obj)
            candidate_types = self._infer_target_room_type(obj)
            logger.info("[VLN] Candidate room types: %s", candidate_types)

            # Check if we're already in the right room
            current_room = self._identify_current_room()
            logger.info("[VLN] Current room: '%s'", current_room)

            accept_any = candidate_types == ["unknown"]
            if not accept_any and current_room not in candidate_types:
                logger.info(
                    "[VLN] Not in target room ('%s' not in %s). Starting room discovery.",
                    current_room, candidate_types,
                )
                discovery_result = self._room_discovery_loop(obj, candidate_types)
                if discovery_result is not None:
                    # Object was spotted during discovery
                    return discovery_result
                if self._search_stop.is_set():
                    return "Search cancelled."
                # _room_discovery_loop returned None → we are now in the target room
                logger.info("[VLN] Room discovery complete. Starting object search.")
            else:
                logger.info(
                    "[VLN] Already in a target room ('%s'). Proceeding to object search.",
                    current_room,
                )

        # Phase 3 — Active object search
        logger.info(f"[VLN] Phase 3: searching for object '{obj}'")

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

        # Third: explore while continuously checking VLM.
        # Delegates to _active_search_loop which handles room-exit recursion
        # when room_exit_enabled is True.
        logger.info(f"[VLN] Starting exploration with active VLM search for '{obj}'")
        found = self._active_search_loop(obj, depth=0)
        if found:
            return found

        if self._search_stop.is_set():
            return "Search cancelled."

        # Build an informative message for the agent on failure
        trail_distance = 0.0
        observed_fraction = 0.0
        try:
            stats_rpc = self.get_rpc_calls("NavDPNavigator.get_exploration_stats")
            stats = stats_rpc()
            trail_distance = stats.get("trail_distance_m", 0.0)
            observed_fraction = stats.get("observed_fraction", 0.0)
        except Exception:
            pass
        return (
            f"Could not find '{obj}'. "
            f"{trail_distance:.1f}m explored, "
            f"{observed_fraction * 100:.0f}% of map observed. "
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

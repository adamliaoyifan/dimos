"""NavDP Navigator — DimOS Module wrapping the diffusion-policy state machine.

Implements ``NavigationInterface`` so DimOS agents / skill containers can
call ``set_goal()`` / ``cancel_goal()`` exactly like the built-in A* planner.

Internal pipeline:
    Inference thread (~6 Hz):
        navdp_image + navdp_depth ─► DiffusionPolicy(HTTP) ─► latest_traj

    Control loop (10 Hz):
        latest_traj ─► TrajectorySelector ─► TrajectoryController ─► cmd_vel
        LiDAR ─► EscapeController (if stuck) ─► override cmd_vel

    The inference thread runs asynchronously so the control loop is never
    blocked waiting for an HTTP round trip.  Each trajectory is valid for
    ~150ms (1-2 ticks) until the next one arrives — the ~4.5cm positional
    drift in that window is within the controller's tolerance.

    On network failure the navigator holds the last computed (v, w) for
    ``hold_duration_s`` then decelerates linearly to zero over
    ``decel_duration_s`` before stopping.  Trajectories are NOT replayed
    across failures because they are in the local frame at inference time
    and become invalid as the robot moves.

State machine (mirroring navdp_bridge_node):
    IDLE → SEEK → APPROACH → STOPPED
              ↑       │
              └───────┘ (lost target)

    SEEK    = nogoal_step  (frontier exploration via diffusion policy)
    APPROACH = imagegoal_step (drive toward captured reference image)
    STOPPED  = zero velocity (depth < threshold)

Dual-camera support:
    navdp_image / navdp_depth — RealSense (or other depth camera) used
        exclusively for trajectory inference and SM depth computation.
    color_image — Go2 built-in camera used for VLM scene detection.
    In simulation both stream names resolve to the same MuJoCo camera.

Requires the NavDP inference server and VLM server running externally.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos_lcm.std_msgs import Bool
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.base import NavigationInterface, NavigationState
from dimos.navigation.navdp.trajectory_selector import TrajectorySelector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy imports from NavDP (pure-Python, no ROS2)
# These are resolved at runtime so the module file can be imported even
# when navdp_bridge is not on PYTHONPATH yet.
# ---------------------------------------------------------------------------
_navdp_bridge = None


def _ensure_navdp_imports() -> None:
    """Import NavDP pure-Python classes once, on first use."""
    global _navdp_bridge
    if _navdp_bridge is not None:
        return
    try:
        from navdp_bridge import state_machine as sm
        from navdp_bridge import goal_context as gc
        from navdp_bridge import escape_controller as esc
        from navdp_bridge import navdp_client as nc
        from navdp_bridge import trajectory_controller as tc
        from navdp_bridge import vlm_client as vc

        _navdp_bridge = {
            "NavState": sm.NavState,
            "NavigationStateMachine": sm.NavigationStateMachine,
            "get_depth_ahead": sm.get_depth_ahead,
            "GoalContext": gc.GoalContext,
            "SearchPhase": gc.SearchPhase,
            "EscapeState": esc.EscapeState,
            "escape_tick": esc.escape_tick,
            "NavDPClient": nc.NavDPClient,
            "TrajectoryController": tc.TrajectoryController,
            "VLMClient": vc.VLMClient,
            "NavigationContext": vc.NavigationContext,
        }
    except ImportError as e:
        logger.error(
            "NavDP bridge package not found. Make sure navdp_bridge is on "
            "PYTHONPATH. Error: %s",
            e,
        )
        raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pose_to_xyyaw(pose: PoseStamped) -> tuple[float, float, float]:
    """Extract (x, y, yaw) from a DimOS PoseStamped."""
    x = pose.position.x
    y = pose.position.y
    # Quaternion → yaw
    q = pose.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    yaw = math.atan2(siny, cosy)
    return (x, y, yaw)


def _image_to_bgr(img: Image) -> np.ndarray:
    """Extract BGR ndarray from a DimOS Image message."""
    data = img.data
    from dimos.msgs.sensor_msgs.Image import ImageFormat

    if img.format == ImageFormat.RGB:
        import cv2

        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return data


# ---------------------------------------------------------------------------
# NavDPNavigator Module
# ---------------------------------------------------------------------------


class NavDPNavigator(Module, NavigationInterface):
    """Diffusion-policy navigation with VLM scene understanding.

    Streams
    -------
    In[Image]        color_image    — Go2 built-in camera (VLM detection only)
    In[Image]        navdp_image    — trajectory camera (RealSense in real, same as
                                      color_image in sim).  Falls back to color_image
                                      if not separately wired.
    In[Image]        depth_image    — depth from the built-in camera (sim fallback)
    In[Image]        navdp_depth    — depth from the trajectory camera (RealSense).
                                      Falls back to depth_image if not wired.
    In[PoseStamped]  odom           — robot odometry
    Out[Twist]       cmd_vel        — velocity commands to robot

    Parameters
    ----------
    navdp_server_url : str
        HTTP endpoint for the NavDP diffusion-policy inference server.
    vlm_server_url : str
        HTTP endpoint for the Qwen3-VL server.
    goal_image_dir : str
        Directory containing reference goal images.
    tick_rate_hz : float
        Control loop frequency (default 10).
    cam_intrinsic : np.ndarray | None
        3x3 camera intrinsic matrix.  If None, uses a reasonable default.
    hold_duration_s : float
        On network failure, hold the last (v, w) for this many seconds before
        beginning deceleration (default 0.3).
    decel_duration_s : float
        After the hold window, decelerate linearly to zero over this many
        seconds (default 0.5).
    enable_internal_vlm : bool
        When False, the navigator's internal VLM detection loop is disabled.
        Use False when VLNSkillContainer drives all scene understanding so the
        two VLM loops do not conflict (default True).
    """

    # --- DimOS streams ---
    color_image: In[Image]
    navdp_image: In[Image]
    depth_image: In[Image]
    navdp_depth: In[Image]
    odom: In[PoseStamped]
    global_costmap: In[OccupancyGrid]
    explore_cmd: In[Bool]
    stop_explore_cmd: In[Bool]
    realsense_camera_info: In[CameraInfo]
    cmd_vel: Out[Twist]
    navdp_path: Out[Path]

    def __init__(
        self,
        navdp_server_url: str = "http://127.0.0.1:8880",
        vlm_server_url: str = "http://127.0.0.1:8866",
        goal_image_dir: str = "/tmp/navdp_goals",
        tick_rate_hz: float = 10.0,
        cam_intrinsic: np.ndarray | None = None,
        # Camera extrinsics (position of camera in base_link frame)
        # Defaults match Unitree Go2 from bridge_params.yaml
        cam_x: float = 0.13,
        cam_y: float = 0.00,
        cam_z: float = 0.30,
        cam_pitch: float = 0.157,  # ~9° downward tilt
        # MPC / trajectory controller parameters
        mpc_horizon: int = 15,
        mpc_desired_v: float = 0.3,
        mpc_v_max: float = 0.3,
        mpc_w_max: float = 0.5,
        mpc_ref_gap: int = 3,
        goal_lookahead_m: float = 1.5,
        # VLM detection interval for state machine
        vlm_detect_interval: float = 1.0,
        # Depth threshold to declare goal reached (metres)
        stop_depth_thresh: float = 0.8,
        # Trajectory selector (costmap-based collision avoidance)
        trajectory_selector_enabled: bool = False,
        trajectory_selector_kwargs: dict | None = None,
        # Network failure graceful handling
        hold_duration_s: float = 0.3,
        decel_duration_s: float = 0.5,
        # VLM control
        enable_internal_vlm: bool = True,
        **kwargs: Any,
    ) -> None:
        self._navdp_url = navdp_server_url
        self._vlm_url = vlm_server_url
        self._goal_image_dir = goal_image_dir
        self._tick_rate_hz = tick_rate_hz
        self._cam_intrinsic = cam_intrinsic
        self._cam_x = cam_x
        self._cam_y = cam_y
        self._cam_z = cam_z
        self._cam_pitch = cam_pitch
        self._mpc_horizon = mpc_horizon
        self._mpc_desired_v = mpc_desired_v
        self._mpc_v_max = mpc_v_max
        self._mpc_w_max = mpc_w_max
        self._mpc_ref_gap = mpc_ref_gap
        self._goal_lookahead_m = goal_lookahead_m
        self._vlm_detect_interval = vlm_detect_interval
        self._stop_depth_thresh = stop_depth_thresh
        self._hold_duration_s = hold_duration_s
        self._decel_duration_s = decel_duration_s
        self._enable_internal_vlm = enable_internal_vlm

        # Live CameraInfo from LcmRealsenseRelay (set when realsense_lcm profile is used)
        self._camera_info_event: threading.Event = threading.Event()
        self._live_camera_info: CameraInfo | None = None

        # Trajectory selector (costmap / LiDAR collision avoidance)
        sel_kwargs = dict(trajectory_selector_kwargs or {})
        sel_kwargs.setdefault("enabled", trajectory_selector_enabled)
        self._trajectory_selector = TrajectorySelector(**sel_kwargs)
        self._latest_costmap: OccupancyGrid | None = None

        # Runtime state (set in start())
        self._state_machine = None
        self._goal_context = None
        self._navdp_client = None
        self._vlm_client = None
        self._traj_ctrl = None
        self._escape_state = None

        # Latest sensor data (protected by _lock)
        self._lock = threading.Lock()
        # color_image stream → VLM detection
        self._latest_image: np.ndarray | None = None
        # depth_image stream → SM depth fallback when navdp_depth not wired
        self._latest_depth: np.ndarray | None = None
        # navdp_image stream → trajectory inference (RealSense in real deployment)
        self._latest_navdp_image: np.ndarray | None = None
        # navdp_depth stream → trajectory inference + SM depth (RealSense)
        self._latest_navdp_depth: np.ndarray | None = None
        self._latest_odom: tuple[float, float, float] | None = None
        self._latest_scan_points: np.ndarray | None = None

        # Async inference state — written by inference thread, read by control tick
        # Protected by _infer_lock (separate from _lock to avoid contention)
        self._infer_lock = threading.Lock()
        self._infer_mode: str = "nogoal"  # "nogoal" or "imagegoal"
        self._infer_ref_img: np.ndarray | None = None
        self._latest_traj: np.ndarray | None = None
        self._latest_all_traj: np.ndarray | None = None
        self._latest_all_vals: np.ndarray | None = None
        self._traj_timestamp: float = 0.0  # time.time() when latest_traj was stored

        # Navigation state
        self._nav_state = NavigationState.IDLE
        self._goal_pose: PoseStamped | None = None
        self._goal_reached = False
        self._running = False
        self._tick_thread: threading.Thread | None = None
        self._inference_thread: threading.Thread | None = None
        self._exploration_enabled = False  # controlled by explore_cmd from web UI
        self._motion_paused = False

        # Language goal for VLM-driven navigation
        self._language_goal: str = ""

        # VLM detection tracking
        self._last_vlm_check: float = 0.0
        self._last_vlm_mode: str = "unknown"
        self._last_vlm_conf: float = 0.0
        self._skip_vlm_detection: bool = False
        # True only on the tick immediately after a fresh VLM result arrives.
        # Prevents stale "unknown" results from being counted 10x/sec by SM.
        self._vlm_result_fresh: bool = False

        # Graceful network failure — hold last good (v,w) then decelerate
        self._last_good_v: float = 0.0
        self._last_good_w: float = 0.0
        self._last_good_time: float = 0.0  # time.time() of last successful inference

        # Visualization rate limiting (avoid flooding Rerun)
        self._last_vis_time: float = 0.0
        self._vis_interval: float = 0.2  # 5 Hz max for debug overlay

        # Diagnostic logging rate limiting
        self._last_diag_time: float = 0.0
        self._diag_interval: float = 5.0  # log status every 5 seconds
        self._inference_fail_count: int = 0
        self._last_infer_mode: str = "unknown"

        # Stuck detection: track recent odom positions to detect when the
        # robot is not making progress (e.g. pushing against a wall).
        self._odom_history: list[tuple[float, float, float, float]] = []  # (t, x, y, yaw)
        self._stuck_window: float = 3.0  # seconds of history to consider
        self._stuck_dist_thresh: float = 0.15  # metres; below this → stuck
        self._selector_halt_streak: int = 0

        # Two-phase escape: backup first, then rotate toward frontier.
        # Phase 0 = not escaping, 1 = backing up, 2 = rotating.
        self._escape_phase: int = 0
        self._escape_phase_start: float = 0.0
        self._escape_backup_duration: float = 1.5    # seconds to back up
        self._escape_backup_speed: float = -0.25     # m/s (negative = reverse)
        self._escape_rotate_duration: float = 2.0    # seconds to rotate
        self._escape_rotate_w: float = 0.5           # rad/s (updated by frontier dir)
        # Cached frontier direction at the moment escape starts, so it
        # doesn't change mid-maneuver.
        self._escape_frontier_dir: tuple[float, float] | None = None
        # Track consecutive escape attempts to escalate rotation aggressiveness
        self._escape_attempt_count: int = 0

        # Global path guidance from hybrid exploration (set via RPC from VLN skill).
        # When set, _get_path_direction() extracts a pure-pursuit direction that
        # replaces the raw frontier-direction hint in the trajectory selector.
        # Protected by _lock (same as costmap / scan_points).
        self._guidance_path: list[tuple[float, float]] | None = None
        self._guidance_path_idx: int = 0

        # Exploration trail: subsampled (x, y) positions for the exploration
        # cost in the trajectory selector.  Updated every tick when the robot
        # has moved >= _explore_trail_sample_dist from the last recorded point.
        self._explore_trail: list[tuple[float, float]] = []
        self._explore_trail_last_pos: tuple[float, float] | None = None
        self._explore_trail_sample_dist: float = 0.5  # metres between samples
        self._explore_trail_max_points: int = 2000

        # Timestamp when navigator last went IDLE (via cancel_goal or goal reached).
        # Used to keep publishing zero cmd_vel for a short braking window.
        self._idle_since: float = 0.0

        # APPROACH grace period: ignore depth-threshold for this many seconds
        # after entering APPROACH, giving the policy time to rotate toward the
        # target before depth_ahead becomes meaningful.
        self._approach_grace_s: float = 2.0
        self._approach_start_time: float = 0.0

        # VLM-driven direction hint for the trajectory selector during APPROACH.
        # Set by VLNSkillContainer via RPC; "left", "centre", "right", or None.
        self._object_direction: str | None = None

        super().__init__(**kwargs)

    @rpc
    def start(self) -> None:
        super().start()
        print("[NavDP] start() called", flush=True)
        try:
            _ensure_navdp_imports()
        except ImportError as e:
            print(f"[NavDP] DISABLED — navdp_bridge import failed: {e}", flush=True)
            logger.error(
                "NavDPNavigator disabled — navdp_bridge not installed. "
                "Add NavDP's ros2_ws/src/navdp_bridge to PYTHONPATH."
            )
            return
        nb = _navdp_bridge

        # Subscribe realsense_camera_info early so the event fires before we wait below
        self._disposables.add(
            Disposable(self.realsense_camera_info.subscribe(self._on_camera_info))
        )

        # Initialize NavDP components — wait for live CameraInfo if no static intrinsic set
        intrinsic = self._cam_intrinsic
        if intrinsic is None:
            logger.info("[NavDP] cam_intrinsic not set — waiting for CameraInfo from relay (up to 30s)...")
            self._camera_info_event.wait(timeout=30.0)
            if self._live_camera_info is not None:
                K = np.array(self._live_camera_info.K, dtype=np.float32).reshape(3, 3)
                intrinsic = K
                logger.info("[NavDP] Using live CameraInfo intrinsic: %s", intrinsic)
            else:
                intrinsic = np.array(
                    [[460, 0, 320], [0, 460, 240], [0, 0, 1]], dtype=np.float32
                )
                logger.warning("[NavDP] CameraInfo timeout — using default intrinsic")
        # Store resolved intrinsic for later reset() calls (e.g. after goal reached)
        self._intrinsic = intrinsic

        # Validate NavDP server reachability before starting
        try:
            import requests as _req
            _req.get(self._navdp_url, timeout=3)
            logger.info("NavDP server reachable at %s", self._navdp_url)
        except Exception as e:
            logger.warning(
                "NavDP server not reachable at %s: %s — will retry during inference",
                self._navdp_url, e,
            )

        self._state_machine = nb["NavigationStateMachine"](
            stop_depth_thresh=self._stop_depth_thresh,
        )
        self._goal_context = nb["GoalContext"](self._goal_image_dir)
        self._navdp_client = nb["NavDPClient"](
            url=self._navdp_url,
        )
        self._navdp_client.initialize(intrinsic)
        self._vlm_client = nb["VLMClient"](
            vlm_url=self._vlm_url,
            logger=logger,
        )
        self._traj_ctrl = nb["TrajectoryController"](
            cam_x=self._cam_x,
            cam_y=self._cam_y,
            cam_z=self._cam_z,
            cam_pitch=self._cam_pitch,
            navdp_rate_hz=self._tick_rate_hz,
            mpc_horizon=self._mpc_horizon,
            mpc_desired_v=self._mpc_desired_v,
            mpc_v_max=self._mpc_v_max,
            mpc_w_max=self._mpc_w_max,
            mpc_ref_gap=self._mpc_ref_gap,
            goal_lookahead_m=self._goal_lookahead_m,
            logger=logger,
        )
        self._escape_state = nb["EscapeState"]()

        # Subscribe to DimOS streams
        print(
            f"[NavDP] stream transports before subscribe: "
            f"color_image={getattr(self.color_image, '_transport', 'MISSING')}, "
            f"navdp_image={getattr(self.navdp_image, '_transport', 'MISSING')}, "
            f"depth_image={getattr(self.depth_image, '_transport', 'MISSING')}, "
            f"navdp_depth={getattr(self.navdp_depth, '_transport', 'MISSING')}, "
            f"odom={getattr(self.odom, '_transport', 'MISSING')}",
            flush=True,
        )
        self._disposables.add(
            Disposable(self.color_image.subscribe(self._on_image))
        )
        self._disposables.add(
            Disposable(self.navdp_image.subscribe(self._on_navdp_image))
        )
        self._disposables.add(
            Disposable(self.depth_image.subscribe(self._on_depth))
        )
        self._disposables.add(
            Disposable(self.navdp_depth.subscribe(self._on_navdp_depth))
        )
        self._disposables.add(
            Disposable(self.odom.subscribe(self._on_odom))
        )
        self._disposables.add(
            Disposable(self.global_costmap.subscribe(self._on_costmap))
        )
        self._disposables.add(
            Disposable(self.explore_cmd.subscribe(self._on_explore_cmd))
        )
        self._disposables.add(
            Disposable(self.stop_explore_cmd.subscribe(self._on_stop_explore_cmd))
        )

        print(
            f"[NavDP] components initialized: sm={type(self._state_machine).__name__}, "
            f"client={type(self._navdp_client).__name__}, "
            f"server={self._navdp_url}, "
            f"trajectory_selector={self._trajectory_selector.enabled}, "
            f"enable_internal_vlm={self._enable_internal_vlm}, "
            f"hold_duration_s={self._hold_duration_s}, "
            f"decel_duration_s={self._decel_duration_s}",
            flush=True,
        )
        logger.info(
            "NavDPNavigator components initialized: state_machine=%s, "
            "navdp_client=%s, vlm_client=%s, traj_ctrl=%s, "
            "enable_internal_vlm=%s",
            type(self._state_machine).__name__,
            type(self._navdp_client).__name__,
            type(self._vlm_client).__name__,
            type(self._traj_ctrl).__name__,
            self._enable_internal_vlm,
        )

        # Start control loop and async inference thread
        self._running = True
        self._tick_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="navdp-control"
        )
        self._tick_thread.start()
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True, name="navdp-inference"
        )
        self._inference_thread.start()
        logger.info("NavDPNavigator started (server=%s)", self._navdp_url)

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=3.0)
        if self._inference_thread is not None:
            self._inference_thread.join(timeout=3.0)
        # Publish zero velocity
        self.cmd_vel.publish(Twist())
        super().stop()

    # --- Stream callbacks ---

    def _on_camera_info(self, info: CameraInfo) -> None:
        """Receive live CameraInfo from LcmRealsenseRelay and unblock start() if waiting."""
        self._live_camera_info = info
        self._camera_info_event.set()

    def _on_image(self, img: Image) -> None:
        """Go2 built-in camera — used for VLM detection only."""
        with self._lock:
            self._latest_image = _image_to_bgr(img)

    def _on_navdp_image(self, img: Image) -> None:
        """Trajectory camera (RealSense in real deployment, same as color_image in sim)."""
        with self._lock:
            self._latest_navdp_image = _image_to_bgr(img)

    def _on_depth(self, img: Image) -> None:
        """Depth from built-in camera — fallback when navdp_depth not wired."""
        with self._lock:
            self._latest_depth = img.data

    def _on_navdp_depth(self, img: Image) -> None:
        """Depth from trajectory camera (RealSense) — used for inference and SM depth."""
        with self._lock:
            self._latest_navdp_depth = img.data

    def _on_odom(self, odom: PoseStamped) -> None:
        if not getattr(self, '_odom_received_logged', False):
            self._odom_received_logged = True
            print(f"[NavDP] _on_odom FIRST callback: pos=({odom.position.x:.3f}, {odom.position.y:.3f})", flush=True)
        with self._lock:
            self._latest_odom = _pose_to_xyyaw(odom)

    def _on_costmap(self, costmap: OccupancyGrid) -> None:
        if not getattr(self, '_costmap_received_logged', False):
            self._costmap_received_logged = True
            print(
                f"[NavDP] _on_costmap FIRST callback: "
                f"{costmap.width}x{costmap.height} @ {costmap.resolution}m",
                flush=True,
            )
        with self._lock:
            self._latest_costmap = costmap

    def _on_explore_cmd(self, msg: Bool) -> None:
        """Handle 'start exploration' command from web UI."""
        self._exploration_enabled = True
        print("[NavDP] exploration ENABLED by web UI", flush=True)
        logger.info("NavDP exploration enabled via explore_cmd")

    def _on_stop_explore_cmd(self, msg: Bool) -> None:
        """Handle 'stop exploration' command from web UI."""
        self._exploration_enabled = False
        # Stop movement immediately
        self.cmd_vel.publish(Twist())
        print("[NavDP] exploration DISABLED by web UI", flush=True)
        logger.info("NavDP exploration disabled via stop_explore_cmd")

    # --- NavigationInterface ---

    @rpc
    def set_goal(self, goal: PoseStamped) -> bool:
        """Accept a pose goal and begin navigating."""
        with self._lock:
            self._goal_pose = goal
            self._goal_reached = False
            self._nav_state = NavigationState.FOLLOWING_PATH
        logger.info(
            "NavDP goal set: (%.2f, %.2f)",
            goal.position.x,
            goal.position.y,
        )
        return True

    @rpc
    def get_state(self) -> NavigationState:
        return self._nav_state

    @rpc
    def is_goal_reached(self) -> bool:
        return self._goal_reached

    @rpc
    def cancel_goal(self) -> bool:
        import traceback as _tb
        print(
            f"[NavDP] cancel_goal() called — setting nav_state→IDLE\n"
            f"  caller: {''.join(_tb.format_stack()[-3:-1]).strip()}",
            flush=True,
        )
        with self._lock:
            self._goal_pose = None
            self._language_goal = ""
            self._nav_state = NavigationState.IDLE
            self._goal_reached = False
        self._odom_history.clear()
        # NOTE: exploration trail is intentionally preserved across
        # cancel_goal so that retried searches benefit from knowing
        # where the robot has already been.  Use clear_exploration_trail()
        # for an explicit reset.
        self._escape_phase = 0
        self._selector_halt_streak = 0
        self._escape_attempt_count = 0
        self._guidance_path = None
        self._guidance_path_idx = 0
        self._skip_vlm_detection = False
        self._object_direction = None
        self._idle_since = time.time()
        if self._state_machine is not None:
            self._state_machine.clear_goal()
        if self._goal_context is not None:
            self._goal_context.clear_language_goal()
        self.cmd_vel.publish(Twist())
        logger.info("NavDP goal cancelled")
        return True

    @rpc
    def pause_motion(self) -> bool:
        """Pause motion without clearing goal/state history."""
        self._motion_paused = True
        self.cmd_vel.publish(Twist())
        logger.info("NavDP motion paused")
        return True

    @rpc
    def resume_motion(self) -> bool:
        """Resume motion after pause_motion()."""
        self._motion_paused = False
        logger.info("NavDP motion resumed")
        return True

    @rpc
    def set_object_direction(self, direction: str) -> None:
        """Set the VLM-detected object direction hint for trajectory selection.

        Args:
            direction: "left", "centre", or "right".
        """
        self._object_direction = direction
        logger.info("NavDP object direction set: %s", direction)

    @rpc
    def clear_object_direction(self) -> None:
        """Clear the object direction hint."""
        self._object_direction = None

    @rpc
    def clear_exploration_trail(self) -> bool:
        """Explicitly clear the exploration trail.

        Unlike ``cancel_goal()`` (which preserves the trail for retries),
        this resets the trail entirely.  Call when starting a fundamentally
        new task where past exploration data is irrelevant.
        """
        self._explore_trail.clear()
        self._explore_trail_last_pos = None
        logger.info("NavDP exploration trail cleared")
        return True

    @rpc
    def get_explore_trail(self) -> list[tuple[float, float]]:
        """Return a snapshot of the exploration trail as (x, y) world-frame points.

        Each point is recorded when the robot moves at least
        ``_explore_trail_sample_dist`` metres from the previous sample (default
        0.5 m).  The list is ordered chronologically — earliest entry first,
        most recent last — so reversing it gives the entry-path backtrack order.

        Returns:
            List of (x, y) tuples in world-frame metres.  Empty list if the
            robot has not yet moved.
        """
        return list(self._explore_trail)

    # --- Hybrid exploration path guidance RPCs ---

    @rpc
    def set_path_guidance(self, waypoints: list[list[float]]) -> bool:
        """Set a global reference path computed by A* for hybrid exploration.

        The path is a list of [x, y] world-frame positions (e.g. from
        ``ReplanningAStarPlanner.compute_path_to_goal``).  During SEEK the
        trajectory selector uses a pure-pursuit direction toward the look-ahead
        point on this path instead of the raw nearest-frontier direction.

        Args:
            waypoints: Ordered list of [x, y] world-frame coordinates.
                       First entry should be near the robot's current position.

        Returns:
            True on success.
        """
        with self._lock:
            self._guidance_path = [(float(w[0]), float(w[1])) for w in waypoints]
            self._guidance_path_idx = 0
        print(
            f"[NavDP] path guidance set: {len(waypoints)} waypoints",
            flush=True,
        )
        return True

    @rpc
    def clear_path_guidance(self) -> bool:
        """Clear the active path guidance (fall back to raw frontier direction)."""
        with self._lock:
            self._guidance_path = None
            self._guidance_path_idx = 0
        return True

    @rpc
    def get_escape_attempt_count(self) -> int:
        """Return the number of consecutive escape attempts since the last free trajectory.

        Used by the VLN skill container to detect dead-end situations and
        trigger an immediate frontier replan without waiting for the normal
        distance-based replan interval.
        """
        return self._escape_attempt_count

    # --- NavDP-specific RPCs ---

    @rpc
    def set_language_goal(self, goal: str) -> bool:
        """Set a language-based navigation goal (e.g., 'find the glasses on desk').

        This triggers VLM-driven hierarchical search via the NavDP state machine.
        The state machine transitions: IDLE → SEEK → APPROACH → STOPPED.

        During SEEK, the diffusion policy uses nogoal_step for exploration.
        When the VLM detects the target, it transitions to APPROACH and uses
        imagegoal_step with a captured reference image to navigate toward it.

        Args:
            goal: Natural language navigation goal.

        Returns:
            True if goal was accepted.
        """
        _ensure_navdp_imports()
        # Detect dummy exploration goals (e.g. VLN skill uses "explore the
        # environment" just to activate nogoal exploration — it runs its own
        # VLM detection externally).  Skip internal VLM detection to prevent
        # false-positive SEEK→APPROACH transitions on generic goals.
        _EXPLORATION_PHRASES = {"explore the environment", "explore", "wander"}
        self._skip_vlm_detection = goal.lower().strip() in _EXPLORATION_PHRASES
        with self._lock:
            self._language_goal = goal
            self._goal_reached = False
            self._nav_state = NavigationState.FOLLOWING_PATH
            self._last_vlm_mode = "unknown"
            self._last_vlm_conf = 0.0
        # Reset stuck detection and path guidance for the new goal
        self._odom_history.clear()
        self._escape_phase = 0
        self._selector_halt_streak = 0
        self._escape_attempt_count = 0
        with self._lock:
            self._guidance_path = None
            self._guidance_path_idx = 0
        if self._goal_context is not None:
            self._goal_context.set_language_goal(goal)
        if self._state_machine is not None:
            self._state_machine.set_goal(has_memory_clue=False)
        print(
            f"[NavDP] set_language_goal({goal!r}) nav_state={self._nav_state} "
            f"sm={self._state_machine.state.name if self._state_machine else 'None'} "
            f"running={self._running} skip_vlm={self._skip_vlm_detection}",
            flush=True,
        )
        logger.info(
            "NavDP language goal: %r (nav_state=%s, sm_state=%s, running=%s)",
            goal, self._nav_state,
            self._state_machine.state.name if self._state_machine else "None",
            self._running,
        )
        return True

    @rpc
    def set_reference_image(self, image_bgr: np.ndarray) -> bool:
        """Set a reference image for APPROACH mode.

        When VLNSkillContainer detects the target via its own VLM, it can
        call this to provide the reference image and force the state machine
        into APPROACH mode.

        Args:
            image_bgr: BGR image array (the frame where the target was seen).

        Returns:
            True if accepted.
        """
        if self._goal_context is None or self._state_machine is None:
            return False
        nb = _navdp_bridge
        # Set reference image on goal context
        ref_img, _ = self._goal_context.set_reference_from_transition(image_bgr)
        # Force state machine to APPROACH
        self._state_machine.state = nb["NavState"].APPROACH
        self._state_machine.lost_count = 0
        # Start grace period so depth-threshold is not checked immediately.
        # The policy needs a moment to rotate toward the target before the
        # centre-ROI depth is meaningful.
        self._approach_start_time = time.time()
        # Skip internal VLM detection — the VLN skill handles detection
        # externally.  Without this, the internal VLM may not recognise the
        # target and revert APPROACH → SEEK via lost_count.
        self._skip_vlm_detection = True
        # Ensure the control loop is active (not skipped by the IDLE guard)
        self._nav_state = NavigationState.FOLLOWING_PATH
        # Tell inference thread to switch to imagegoal_step
        with self._infer_lock:
            self._infer_mode = "imagegoal"
            self._infer_ref_img = ref_img
        logger.info("NavDP reference image set externally → APPROACH (nav_state=FOLLOWING_PATH, skip_vlm=True)")
        return True

    @rpc
    def get_language_goal(self) -> str:
        """Return the current language goal, or empty string if none."""
        return self._language_goal

    @rpc
    def get_navdp_state(self) -> str:
        """Return the NavDP state machine state name (IDLE/RECALL/NAVIGATE/...)."""
        if self._state_machine is None:
            return "UNINITIALIZED"
        return self._state_machine.state.name

    @rpc
    def get_exploration_stats(self) -> dict:
        """Return exploration coverage statistics for adaptive search timeout.

        Returns a dict with:
        - trail_length: number of trail points recorded
        - trail_distance_m: total path length of the trail (metres)
        - coverage_radius_m: max distance from trail centroid (rough coverage radius)
        - observed_fraction: fraction of costmap cells that are observed (not UNKNOWN)
        - costmap_available: whether a costmap is currently available
        """
        trail = list(self._explore_trail)
        costmap = self._latest_costmap

        # Trail distance (arc length)
        trail_distance = 0.0
        if len(trail) >= 2:
            for i in range(1, len(trail)):
                trail_distance += math.hypot(
                    trail[i][0] - trail[i - 1][0],
                    trail[i][1] - trail[i - 1][1],
                )

        # Coverage radius (max distance from centroid)
        coverage_radius = 0.0
        if len(trail) >= 2:
            xs = [p[0] for p in trail]
            ys = [p[1] for p in trail]
            cx, cy = sum(xs) / len(xs), sum(ys) / len(ys)
            coverage_radius = max(
                math.hypot(x - cx, y - cy) for x, y in trail
            )

        # Costmap observed fraction
        observed_fraction = 0.0
        costmap_available = costmap is not None
        if costmap is not None:
            grid = costmap.grid
            n_total = grid.size
            if n_total > 0:
                n_observed = int(np.count_nonzero(grid != CostValues.UNKNOWN))
                observed_fraction = n_observed / n_total

        return {
            "trail_length": len(trail),
            "trail_distance_m": round(trail_distance, 2),
            "coverage_radius_m": round(coverage_radius, 2),
            "observed_fraction": round(observed_fraction, 4),
            "costmap_available": costmap_available,
        }

    # --- Stuck detection ---

    def _check_stuck(self, odom: tuple[float, float, float]) -> bool:
        """Return True if the robot has not moved enough over the last few seconds.

        Uses a sliding window of odom positions.  When the total displacement
        over ``_stuck_window`` seconds is below ``_stuck_dist_thresh``, the
        robot is considered stuck.
        """
        now = time.time()
        x, y, yaw = odom
        self._odom_history.append((now, x, y, yaw))

        # Prune entries older than the window
        cutoff = now - self._stuck_window
        self._odom_history = [
            e for e in self._odom_history if e[0] >= cutoff
        ]

        # Need at least a full window of data to decide
        if not self._odom_history or (now - self._odom_history[0][0]) < self._stuck_window * 0.9:
            return False

        oldest = self._odom_history[0]
        dist = math.hypot(x - oldest[1], y - oldest[2])
        stuck = dist < self._stuck_dist_thresh
        yaw_delta = abs((yaw - oldest[3] + math.pi) % (2 * math.pi) - math.pi)
        if self._selector_halt_streak >= 3 or stuck or dist < (self._stuck_dist_thresh * 3.0):
            last_diag = getattr(self, "_last_stuck_debug_time", 0.0)
            if (now - last_diag) > 0.75:
                self._last_stuck_debug_time = now
        return stuck

    # --- Exploration trail ---

    def _update_explore_trail(self, x: float, y: float) -> None:
        """Record the robot's position if it moved far enough from the last sample.

        The trail is used by the trajectory selector to penalise trajectories
        heading into already-explored areas during SEEK state.
        """
        if self._explore_trail_last_pos is None:
            self._explore_trail.append((x, y))
            self._explore_trail_last_pos = (x, y)
            return
        lx, ly = self._explore_trail_last_pos
        if math.hypot(x - lx, y - ly) >= self._explore_trail_sample_dist:
            self._explore_trail.append((x, y))
            self._explore_trail_last_pos = (x, y)
            if len(self._explore_trail) > self._explore_trail_max_points:
                self._explore_trail = self._explore_trail[-self._explore_trail_max_points:]

    # --- Frontier direction ---

    def _get_nearest_frontier_direction(
        self,
        odom: tuple[float, float, float],
        costmap: OccupancyGrid | None,
        search_radius_m: float = 5.0,
    ) -> tuple[float, float] | None:
        """Compute a unit vector from the robot toward the nearest frontier centroid.

        A *frontier cell* is a FREE cell that has at least one UNKNOWN
        neighbour (4-connected).  We search a square patch of the costmap
        centred on the robot (side = 2 * search_radius_m) and return the
        direction toward the centroid of the closest cluster of frontier
        cells.

        Returns ``(dx, dy)`` unit vector in world frame, or ``None`` if no
        frontier is found within the search radius.
        """
        if costmap is None:
            return None

        grid = costmap.grid
        res = costmap.resolution
        if res <= 0 or grid.size == 0:
            return None

        ox, oy, _ = odom
        robot_grid = costmap.world_to_grid((ox, oy, 0.0))
        rgx, rgy = int(robot_grid.x), int(robot_grid.y)

        h, w = grid.shape
        radius_cells = int(math.ceil(search_radius_m / res))

        # Clip search box to grid bounds
        x_min = max(0, rgx - radius_cells)
        x_max = min(w - 1, rgx + radius_cells)
        y_min = max(0, rgy - radius_cells)
        y_max = min(h - 1, rgy + radius_cells)
        if x_min >= x_max or y_min >= y_max:
            return None

        patch = grid[y_min:y_max + 1, x_min:x_max + 1]
        ph, pw = patch.shape

        # Build FREE mask and UNKNOWN mask
        free_mask = patch == CostValues.FREE
        unknown_mask = patch == CostValues.UNKNOWN

        # Frontier = FREE cell with at least one UNKNOWN 4-neighbour
        frontier_mask = np.zeros_like(free_mask)
        if ph > 1:
            frontier_mask[1:, :] |= free_mask[1:, :] & unknown_mask[:-1, :]   # up
            frontier_mask[:-1, :] |= free_mask[:-1, :] & unknown_mask[1:, :]  # down
        if pw > 1:
            frontier_mask[:, 1:] |= free_mask[:, 1:] & unknown_mask[:, :-1]   # left
            frontier_mask[:, :-1] |= free_mask[:, :-1] & unknown_mask[:, 1:]  # right

        frontier_ys, frontier_xs = np.where(frontier_mask)
        if len(frontier_xs) == 0:
            return None

        # Convert frontier cells back to world coordinates
        world_xs = (frontier_xs + x_min) * res + costmap.origin.position.x
        world_ys = (frontier_ys + y_min) * res + costmap.origin.position.y

        # Find the N closest frontier cells and use their centroid
        dists_sq = (world_xs - ox) ** 2 + (world_ys - oy) ** 2
        n_closest = min(50, len(dists_sq))
        closest_idx = np.argpartition(dists_sq, n_closest)[:n_closest]

        cx = float(world_xs[closest_idx].mean())
        cy = float(world_ys[closest_idx].mean())

        dx = cx - ox
        dy = cy - oy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-6:
            return None

        return (dx / dist, dy / dist)

    def _get_path_direction(
        self,
        odom: tuple[float, float, float],
        lookahead_m: float = 2.0,
    ) -> tuple[float, float] | None:
        """Return a unit vector from the robot toward the look-ahead point on the guidance path.

        Implements pure-pursuit style look-ahead: advances the tracked index to
        the closest waypoint ahead, then walks forward along the path until
        ``lookahead_m`` cumulative distance is covered and returns the direction
        to that target point.

        Returns ``(dx, dy)`` world-frame unit vector, or ``None`` when no
        guidance path is set or the path has been exhausted.
        """
        with self._lock:
            path = self._guidance_path
            idx = self._guidance_path_idx

        if path is None or len(path) < 2:
            return None

        ox, oy, _ = odom

        # Advance index to the closest waypoint within the next 20 entries,
        # ensuring the tracked point stays near the robot.
        search_end = min(idx + 20, len(path))
        best_idx = idx
        best_dist = float("inf")
        for i in range(idx, search_end):
            d = math.hypot(path[i][0] - ox, path[i][1] - oy)
            if d < best_dist:
                best_dist = d
                best_idx = i
        with self._lock:
            self._guidance_path_idx = best_idx

        # Walk forward from best_idx until we accumulate lookahead_m distance
        cum_dist = 0.0
        target_idx = best_idx
        for i in range(best_idx, len(path) - 1):
            seg = math.hypot(
                path[i + 1][0] - path[i][0],
                path[i + 1][1] - path[i][1],
            )
            cum_dist += seg
            target_idx = i + 1
            if cum_dist >= lookahead_m:
                break

        tx, ty = path[target_idx]
        dx, dy = tx - ox, ty - oy
        mag = math.hypot(dx, dy)
        if mag < 1e-6:
            return None
        return (dx / mag, dy / mag)

    # --- VLM detection (background, periodic) ---

    def _run_vlm_detection(self, image: np.ndarray) -> None:
        """Run VLM detection query if enough time has passed."""
        now = time.time()
        if now - self._last_vlm_check < self._vlm_detect_interval:
            return
        self._last_vlm_check = now

        goal = self._language_goal
        if not goal or self._vlm_client is None:
            return

        try:
            result = self._vlm_client.query_detection(image, goal)
            if result:
                self._last_vlm_mode = result.get("mode", "unknown")
                self._last_vlm_conf = result.get("confidence", 0.0)
                self._vlm_result_fresh = True
        except Exception:
            # Rate-limited warning so repeated failures are visible in logs
            # without flooding at 1 Hz.
            _now = time.time()
            if _now - getattr(self, "_last_vlm_fail_log", 0.0) > 10.0:
                self._last_vlm_fail_log = _now
                logger.warning("VLM detection query failed (last 10s)", exc_info=True)

    # --- Visualization ---

    def _publish_debug_vis(
        self,
        all_traj: np.ndarray | None,
        all_vals: np.ndarray | None,
        waypoints: np.ndarray | None,
        odom: tuple[float, float, float],
        v: float,
        w: float,
    ) -> None:
        """Publish NavDP debug visualization to Rerun and navdp_path stream.

        Renders:
        - 16 candidate trajectories as semi-transparent colored lines
        - Selected trajectory highlighted in bright cyan
        - State machine state, VLM info, and method label as text
        """
        now = time.time()
        if now - self._last_vis_time < self._vis_interval:
            return
        self._last_vis_time = now

        # --- Publish selected trajectory as Path (for WebSocket 2D vis) ---
        if waypoints is not None and len(waypoints) > 0:
            ox, oy, oyaw = odom
            cos_yaw = math.cos(oyaw)
            sin_yaw = math.sin(oyaw)
            poses = []
            for wp in waypoints:
                # Transform base_link waypoints to world frame
                wx = ox + cos_yaw * wp[0] - sin_yaw * wp[1]
                wy = oy + sin_yaw * wp[0] + cos_yaw * wp[1]
                poses.append(PoseStamped(position=[wx, wy, 0.0]))
            self.navdp_path.publish(Path(poses=poses, frame_id="world"))

        # --- Rerun 3D visualization ---
        try:
            import rerun as rr
        except ImportError:
            return

        sm_state = "UNINITIALIZED"
        if self._state_machine is not None:
            sm_state = self._state_machine.state.name

        # Candidate trajectories (all 16)
        if all_traj is not None and self._traj_ctrl is not None:
            candidate_strips = []
            candidate_colors = []
            n_candidates = all_traj.shape[0] if all_traj.ndim == 3 else 1

            # Normalize scores for color mapping
            scores = all_vals.flatten() if all_vals is not None else np.zeros(n_candidates)
            s_min, s_max = scores.min(), scores.max()
            s_range = s_max - s_min if s_max > s_min else 1.0

            for i in range(n_candidates):
                t = all_traj[i] if all_traj.ndim == 3 else all_traj
                wp = self._traj_ctrl.trajectory_to_waypoints(t)
                if len(wp) == 0:
                    continue
                # Transform to world frame
                ox, oy, oyaw = odom
                cos_yaw = math.cos(oyaw)
                sin_yaw = math.sin(oyaw)
                world_pts = []
                for p in wp:
                    wx = ox + cos_yaw * p[0] - sin_yaw * p[1]
                    wy = oy + sin_yaw * p[0] + cos_yaw * p[1]
                    world_pts.append([wx, wy, 0.6])  # z=0.6 above floor
                candidate_strips.append(world_pts)
                # Color: low score = red, high score = green
                norm = (scores[i] - s_min) / s_range if i < len(scores) else 0.5
                r = int(255 * (1.0 - norm))
                g = int(255 * norm)
                candidate_colors.append([r, g, 50, 100])  # semi-transparent

            if candidate_strips:
                rr.log(
                    "world/navdp/candidates",
                    rr.LineStrips3D(
                        candidate_strips,
                        colors=candidate_colors,
                        radii=0.02,
                    ),
                )

        # Selected trajectory (bright cyan, thicker)
        if waypoints is not None and len(waypoints) > 0:
            ox, oy, oyaw = odom
            cos_yaw = math.cos(oyaw)
            sin_yaw = math.sin(oyaw)
            selected_pts = []
            for wp in waypoints:
                wx = ox + cos_yaw * wp[0] - sin_yaw * wp[1]
                wy = oy + sin_yaw * wp[0] + cos_yaw * wp[1]
                selected_pts.append([wx, wy, 0.7])  # slightly above candidates
            rr.log(
                "world/navdp/selected",
                rr.LineStrips3D(
                    [selected_pts],
                    colors=[[0, 255, 255, 255]],  # bright cyan
                    radii=0.05,
                ),
            )

        # State machine info + method label
        state_text = (
            f"[NavDP] State: {sm_state}\n"
            f"Goal: {self._language_goal or '(none)'}\n"
            f"VLM: {self._last_vlm_mode} (conf={self._last_vlm_conf:.2f})\n"
            f"Cmd: v={v:.2f} w={w:.2f}\n"
            f"Method: NavDP (diffusion-policy)"
        )
        rr.log("world/navdp/state_info", rr.TextLog(state_text, level="INFO"))

        # 3D text annotation at robot position
        rr.log(
            "world/navdp/method_label",
            rr.Points3D(
                [[odom[0], odom[1], 1.2]],
                labels=[f"NavDP | {sm_state}"],
                colors=[[0, 255, 255, 200]],
                radii=0.08,
            ),
        )

    # --- Control loop ---

    def _inference_loop(self) -> None:
        """Background thread — sends image+depth to NavDP server at maximum rate.

        Runs independently of the control loop so HTTP latency (~150ms) never
        blocks cmd_vel publishing.  Results are stored under _infer_lock and
        read by _tick() on the next control iteration.

        The thread reads the current inference mode (nogoal vs imagegoal) and
        reference image from _infer_mode/_infer_ref_img, which the control
        thread updates on SM transitions.  The inference image/depth come from
        navdp_image/navdp_depth (RealSense) with fallback to color_image/depth_image.
        """
        infer_count = 0
        while self._running:
            # Bail out immediately if core components are not yet initialised
            if self._navdp_client is None or self._traj_ctrl is None:
                time.sleep(0.05)
                continue

            nb = _navdp_bridge
            if nb is None:
                time.sleep(0.05)
                continue

            NavState = nb["NavState"]

            # --- Grab latest sensor data ---
            with self._lock:
                # Prefer dedicated navdp streams; fall back to shared camera streams
                image = (
                    self._latest_navdp_image
                    if self._latest_navdp_image is not None
                    else self._latest_image
                )
                depth_raw = (
                    self._latest_navdp_depth
                    if self._latest_navdp_depth is not None
                    else self._latest_depth
                )

            if image is None:
                time.sleep(0.02)
                continue

            # Build depth plane
            if depth_raw is not None and depth_raw.shape[:2] == image.shape[:2]:
                depth = depth_raw.astype(np.float32) if depth_raw.dtype != np.float32 else depth_raw
            else:
                depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

            # --- Read current inference mode from control thread ---
            with self._infer_lock:
                mode = self._infer_mode
                ref_img = self._infer_ref_img

            # --- Call NavDP server ---
            traj = None
            all_traj = None
            all_vals = None
            t_inf = time.time()
            try:
                if mode == "imagegoal" and ref_img is not None:
                    traj, all_traj, all_vals = self._navdp_client.imagegoal_step(
                        image, depth, ref_img
                    )
                else:
                    traj, all_traj, all_vals = self._navdp_client.nogoal_step(
                        image, depth
                    )
                inf_ms = (time.time() - t_inf) * 1000

                if infer_count == 0:
                    _vals_range = (
                        f"[{float(all_vals.min()):.3f}, {float(all_vals.max()):.3f}]"
                        if all_vals is not None else "None"
                    )
                    print(
                        f"[NavDP] FIRST inference result: mode={mode}, "
                        f"traj={traj.shape if traj is not None else None}, "
                        f"all_traj={all_traj.shape if all_traj is not None else None}, "
                        f"all_vals_range={_vals_range}, "
                        f"time={inf_ms:.0f}ms, server={self._navdp_url}",
                        flush=True,
                    )
                elif infer_count % 50 == 0:
                    print(
                        f"[NavDP] inference #{infer_count}: mode={mode}, "
                        f"traj={'OK' if traj is not None else 'None'}, "
                        f"time={inf_ms:.0f}ms",
                        flush=True,
                    )
                infer_count += 1

            except Exception:
                logger.warning(
                    "NavDP inference failed (mode=%s, server=%s)",
                    mode, self._navdp_url, exc_info=True,
                )
                # Do not update _latest_traj — let the control tick handle staleness
                continue

            if traj is None:
                logger.debug("NavDP inference returned None trajectory (mode=%s)", mode)
                continue

            # Squeeze leading batch dimension: server returns (1, 24, 3) → (24, 3)
            if traj.ndim == 3 and traj.shape[0] == 1:
                traj = traj[0]

            # --- Store results for control tick ---
            with self._infer_lock:
                self._latest_traj = traj
                self._latest_all_traj = all_traj
                self._latest_all_vals = all_vals
                self._traj_timestamp = time.time()

    def _control_loop(self) -> None:
        """Main tick loop — publishes cmd_vel at tick_rate_hz using latest inference."""
        dt = 1.0 / self._tick_rate_hz
        tick_count = 0
        while self._running:
            t0 = time.time()
            try:
                self._tick()
            except Exception as e:
                import traceback
                print(f"[NavDP] TICK ERROR: {e}\n{traceback.format_exc()}", flush=True)
                logger.exception("NavDP control tick error")
            tick_count += 1
            elapsed_ms = (time.time() - t0) * 1000
            if tick_count % 50 == 0:
                print(
                    f"[NavDP] tick #{tick_count}  nav={self._nav_state.name}  "
                    f"sm={self._state_machine.state.name if self._state_machine else 'N/A'}  "
                    f"elapsed={elapsed_ms:.0f}ms  "
                    f"fails={self._inference_fail_count}",
                    flush=True,
                )
            time.sleep(dt)

    # ------------------------------------------------------------------
    # Control-loop context (passed between _tick_* stage methods)
    # ------------------------------------------------------------------

    @dataclass
    class _TickContext:
        """Data bag passed between _tick_* stage methods each control cycle."""
        image: np.ndarray
        odom: "tuple[float, float, float]"
        # Populated by _tick_state_machine_update
        sm_depth: np.ndarray = field(
            default_factory=lambda: np.zeros((1, 1), dtype=np.float32)
        )
        depth_ahead: float = 0.0
        # Populated by _tick_read_trajectory
        traj: "np.ndarray | None" = None
        all_traj: "np.ndarray | None" = None
        all_vals: "np.ndarray | None" = None
        traj_age: float = float("inf")
        infer_mode: str = "nogoal"
        # Populated by _tick_select_trajectory
        waypoints: "np.ndarray | None" = None
        v: float = 0.0
        w: float = 0.0

    # ------------------------------------------------------------------
    # _tick stage methods
    # ------------------------------------------------------------------

    def _tick_guard_checks(self) -> "_TickContext | None":
        """Acquire sensor snapshot; early-return for unready / paused / IDLE state.

        Returns a populated _TickContext or None if the tick should exit.
        """
        with self._lock:
            image = self._latest_image
            odom = self._latest_odom

        if image is None or odom is None:
            now = time.time()
            if now - self._last_diag_time > self._diag_interval:
                self._last_diag_time = now
                print(
                    f"[NavDP] tick: waiting for sensor data "
                    f"(image={'yes' if image is not None else 'NO'}, "
                    f"odom={'yes' if odom is not None else 'NO'})",
                    flush=True,
                )
            return None

        # When IDLE and exploration not enabled from the web UI, skip tick.
        # Keep publishing zero cmd_vel for a short braking window after cancel
        # so MuJoCo simulation doesn't coast on the last velocity command.
        if self._nav_state == NavigationState.IDLE and not self._exploration_enabled:
            if (time.time() - self._idle_since) < 1.0:
                self.cmd_vel.publish(Twist())
            return None

        # Transient pause used by VLN blocking VLM checks.  Unlike cancel_goal(),
        # this intentionally preserves nav state and stuck/escape history.
        if self._motion_paused:
            self.cmd_vel.publish(Twist())
            return None

        # Record position for exploration trail (used by trajectory selector)
        self._update_explore_trail(odom[0], odom[1])

        # Log first tick after goal is set
        if not hasattr(self, "_first_active_tick_logged"):
            self._first_active_tick_logged = False
        if not self._first_active_tick_logged:
            self._first_active_tick_logged = True
            print(
                f"[NavDP] first active tick: nav_state={self._nav_state}, "
                f"image={image.shape}, odom={odom}",
                flush=True,
            )

        if _navdp_bridge is None:
            print("[NavDP] tick: navdp_bridge=None, skipping", flush=True)
            return None

        return self._TickContext(image=image, odom=odom)

    def _tick_vlm_detection(self, ctx: "_TickContext") -> None:
        """Run periodic VLM detection for state machine SEEK/APPROACH transitions."""
        nb = _navdp_bridge
        NavState = nb["NavState"]
        if (
            self._enable_internal_vlm
            and self._language_goal
            and self._state_machine is not None
            and not self._skip_vlm_detection
        ):
            sm_state = self._state_machine.state
            if sm_state in (NavState.SEEK, NavState.APPROACH):
                self._run_vlm_detection(ctx.image)

    def _tick_state_machine_update(self, ctx: "_TickContext") -> bool:
        """Compute depth, update SM, handle state transitions.

        Populates ctx.sm_depth and ctx.depth_ahead.
        Returns True if the tick should exit early (goal reached → STOPPED).
        """
        nb = _navdp_bridge
        NavState = nb["NavState"]

        # --- Compute depth for state machine (navdp_depth preferred) ---
        # Use the trajectory camera's depth for SM transitions (more accurate).
        # Falls back to the built-in camera depth, then zeros if neither available.
        with self._lock:
            navdp_depth_raw = self._latest_navdp_depth
            fallback_depth_raw = self._latest_depth

        sm_depth_raw = navdp_depth_raw if navdp_depth_raw is not None else fallback_depth_raw
        if sm_depth_raw is not None and sm_depth_raw.shape[:2] == ctx.image.shape[:2]:
            ctx.sm_depth = (
                sm_depth_raw.astype(np.float32)
                if sm_depth_raw.dtype != np.float32
                else sm_depth_raw
            )
        else:
            ctx.sm_depth = np.zeros(
                (ctx.image.shape[0], ctx.image.shape[1]), dtype=np.float32
            )
        ctx.depth_ahead = nb["get_depth_ahead"](ctx.sm_depth)

        if self._state_machine is None:
            return False

        old_state = self._state_machine.state
        # Only pass a real VLM result to the state machine on the tick where
        # the VLM actually ran.  On all other ticks use "pending" so the SM
        # does not count stale "unknown" results against lost_count (which
        # would fire at 10 Hz, abandoning APPROACH after only ~0.5 s).
        vlm_mode_for_sm = self._last_vlm_mode if self._vlm_result_fresh else "pending"
        vlm_conf_for_sm = self._last_vlm_conf if self._vlm_result_fresh else 0.0
        self._vlm_result_fresh = False

        # During the APPROACH grace period the centre-ROI depth reading may
        # reflect a nearby surface the robot is currently facing (e.g. a wall
        # behind the target).  Clamping to inf prevents a premature
        # APPROACH → STOPPED transition before the policy has had time to
        # rotate and advance toward the target.
        approach_age = time.time() - self._approach_start_time
        effective_depth = (
            float("inf")
            if approach_age < self._approach_grace_s
            else ctx.depth_ahead
        )

        self._state_machine.update(
            vlm_mode=vlm_mode_for_sm,
            vlm_conf=vlm_conf_for_sm,
            depth_ahead=effective_depth,
        )
        new_state = self._state_machine.state

        if old_state != new_state:
            print(
                f"[NavDP] SM transition: {old_state.name} → {new_state.name}  "
                f"vlm={self._last_vlm_mode}(conf={self._last_vlm_conf:.2f})  "
                f"depth_ahead={ctx.depth_ahead:.2f} (effective={effective_depth:.2f})",
                flush=True,
            )

        # SEEK → APPROACH: target detected — capture reference image and
        # tell the inference thread to switch to imagegoal_step.
        if old_state == NavState.SEEK and new_state == NavState.APPROACH:
            ref_img_cap, image_src = self._goal_context.set_reference_from_transition(
                ctx.image
            )
            # Start grace period so depth-threshold is not checked immediately.
            self._approach_start_time = time.time()
            with self._infer_lock:
                self._infer_mode = "imagegoal"
                self._infer_ref_img = ref_img_cap
            logger.info(
                "NavDP: target detected (conf=%.2f), using %s → APPROACH",
                self._last_vlm_conf,
                image_src,
            )

        # APPROACH → STOPPED: goal reached — stop and reset.
        if old_state == NavState.APPROACH and new_state == NavState.STOPPED:
            print(f"[NavDP] GOAL REACHED → IDLE (depth={ctx.depth_ahead:.2f})", flush=True)
            logger.info("NavDP: goal reached (depth=%.2f) → STOPPED", ctx.depth_ahead)
            self._goal_reached = True
            self._nav_state = NavigationState.IDLE
            self._object_direction = None
            self._idle_since = time.time()
            self._state_machine.reset()
            with self._infer_lock:
                self._infer_mode = "nogoal"
                self._infer_ref_img = None
            self.cmd_vel.publish(Twist())
            return True  # exit tick

        # APPROACH → SEEK: target lost — revert inference to nogoal exploration.
        if old_state == NavState.APPROACH and new_state == NavState.SEEK:
            with self._infer_lock:
                self._infer_mode = "nogoal"
                self._infer_ref_img = None
            logger.info("NavDP: target lost in APPROACH → SEEK (nogoal)")

        return False

    def _tick_read_trajectory(self, ctx: "_TickContext") -> bool:
        """Read the latest trajectory from the inference thread.

        Populates ctx.traj, ctx.all_traj, ctx.all_vals, ctx.traj_age,
        ctx.infer_mode.

        If the trajectory is missing or stale, applies hold/decel/stop logic,
        publishes cmd_vel, and returns False (tick done).  Returns True when a
        fresh trajectory is available and the tick should continue.
        """
        nb = _navdp_bridge
        NavState = nb["NavState"]

        with self._infer_lock:
            ctx.traj = self._latest_traj
            ctx.all_traj = self._latest_all_traj
            ctx.all_vals = self._latest_all_vals
            ctx.traj_age = (
                time.time() - self._traj_timestamp
                if self._traj_timestamp > 0
                else float("inf")
            )
            ctx.infer_mode = self._infer_mode
        self._last_infer_mode = ctx.infer_mode

        sm_state = self._state_machine.state if self._state_machine else NavState.SEEK

        # --- Handle missing or stale trajectory (graceful network failure) ---
        if ctx.traj is None or ctx.traj_age > (
            self._hold_duration_s + self._decel_duration_s
        ):
            now = time.time()
            age_since_good = (
                now - self._last_good_time if self._last_good_time > 0 else float("inf")
            )
            if age_since_good <= self._hold_duration_s:
                ctx.v = self._last_good_v
                ctx.w = self._last_good_w
                if now - self._last_diag_time > self._diag_interval:
                    self._last_diag_time = now
                    logger.warning(
                        "NavDP: no fresh trajectory (age=%.2fs) — holding v=%.2f w=%.2f",
                        ctx.traj_age,
                        ctx.v,
                        ctx.w,
                    )
            elif age_since_good <= self._hold_duration_s + self._decel_duration_s:
                elapsed_decel = age_since_good - self._hold_duration_s
                scale = max(0.0, 1.0 - elapsed_decel / self._decel_duration_s)
                ctx.v = self._last_good_v * scale
                ctx.w = self._last_good_w * scale
                if now - self._last_diag_time > self._diag_interval:
                    self._last_diag_time = now
                    logger.warning(
                        "NavDP: no fresh trajectory (age=%.2fs) — decelerating scale=%.2f",
                        ctx.traj_age,
                        scale,
                    )
            else:
                ctx.v = 0.0
                ctx.w = 0.0
                self._inference_fail_count += 1
                if now - self._last_diag_time > self._diag_interval:
                    self._last_diag_time = now
                    print(
                        f"[NavDP] no fresh trajectory for {ctx.traj_age:.1f}s "
                        f"({self._inference_fail_count} consecutive). "
                        f"sm={sm_state.name if hasattr(sm_state, 'name') else sm_state}, "
                        f"server={self._navdp_url}",
                        flush=True,
                    )
                    logger.warning(
                        "NavDP: no fresh trajectory (%d consecutive). "
                        "State=%s, server=%s. Is the NavDP server running?",
                        self._inference_fail_count,
                        sm_state.name if hasattr(sm_state, "name") else sm_state,
                        self._navdp_url,
                    )
            self.cmd_vel.publish(
                Twist(
                    linear=Vector3(x=float(ctx.v), y=0.0, z=0.0),
                    angular=Vector3(x=0.0, y=0.0, z=float(ctx.w)),
                )
            )
            return False  # trajectory handled, tick done

        self._inference_fail_count = 0

        # --- Log trajectory diagnostics ---
        now = time.time()
        if now - self._last_diag_time > self._diag_interval:
            self._last_diag_time = now
            n_candidates = (
                ctx.all_traj.shape[0]
                if ctx.all_traj is not None and ctx.all_traj.ndim == 3
                else 0
            )
            best_val = (
                float(ctx.all_vals.max())
                if ctx.all_vals is not None and ctx.all_vals.size > 0
                else 0.0
            )
            logger.info(
                "NavDP trajectory: mode=%s, selected_shape=%s, "
                "candidates=%d, best_score=%.3f, traj_range=[%.3f, %.3f], age=%.0fms",
                ctx.infer_mode,
                ctx.traj.shape,
                n_candidates,
                best_val,
                float(ctx.traj.min()),
                float(ctx.traj.max()),
                ctx.traj_age * 1000,
            )

        return True

    def _tick_select_trajectory(self, ctx: "_TickContext") -> None:
        """Run the trajectory selector (when enabled) and convert to (v, w).

        Populates ctx.waypoints, ctx.v, ctx.w.
        """
        traj = ctx.traj  # may be replaced by selector result

        # --- Trajectory selection (costmap / LiDAR collision avoidance) ---
        if (
            self._trajectory_selector.enabled
            and ctx.all_traj is not None
            and ctx.odom is not None
        ):
            if not getattr(self, "_traj_sel_logged", False):
                self._traj_sel_logged = True
                print(
                    f"[NavDP] TrajectorySelector ACTIVE: "
                    f"all_traj={ctx.all_traj.shape}, "
                    f"costmap={'yes' if self._latest_costmap is not None else 'NO'}, "
                    f"scan_pts={'yes' if self._latest_scan_points is not None else 'NO'}",
                    flush=True,
                )
            with self._lock:
                costmap = self._latest_costmap
                scan_pts = self._latest_scan_points

            # Build explored-positions array for exploration cost (SEEK only).
            _NavState = _navdp_bridge["NavState"] if _navdp_bridge else None
            _is_seeking = (
                _NavState is not None
                and self._state_machine is not None
                and self._state_machine.state == _NavState.SEEK
            )
            _explored_pos = (
                np.array(self._explore_trail, dtype=np.float32)
                if _is_seeking and self._explore_trail
                else None
            )

            # Compute directional hint for SEEK — prefer A* path guidance when
            # available (hybrid mode), fall back to nearest frontier direction.
            if _is_seeking:
                _path_dir = self._get_path_direction(ctx.odom)
                _frontier_dir = (
                    _path_dir
                    if _path_dir is not None
                    else self._get_nearest_frontier_direction(ctx.odom, costmap)
                )
            else:
                _frontier_dir = None

            sel_result = self._trajectory_selector.select(
                selected_traj=traj,
                all_trajectories=ctx.all_traj,
                all_values=ctx.all_vals,
                odom=ctx.odom,
                traj_to_waypoints_fn=self._traj_ctrl.trajectory_to_waypoints,
                costmap=costmap,
                scan_points=scan_pts,
                explored_positions=_explored_pos,
                is_seeking=_is_seeking,
                depth_image=ctx.sm_depth,
                object_direction=self._object_direction,
                frontier_direction=_frontier_dir,
            )
            # Stash for downstream diagnostic (publish-time log).
            self._last_sel_fallback = bool(sel_result.fallback_used)
            self._last_sel_index = int(sel_result.index)
            self._last_sel_cost = float(sel_result.cost)
            if sel_result.fallback_used:
                self._selector_halt_streak += 1
                # All trajectories collide — zero velocity but do NOT return
                # early so stuck detection and escape rotation can still fire.
                ctx.v = 0.0
                ctx.w = 0.0
                ctx.waypoints = np.zeros((0, 2), dtype=np.float32)
            else:
                # Decrement rather than reset — a single free trajectory amid
                # many all-collide ticks should NOT fully clear the streak.
                # This prevents the robot from slowly creeping into walls when
                # only 1/8 trajectories is occasionally collision-free.
                self._selector_halt_streak = max(0, self._selector_halt_streak - 2)
                # Once escape has succeeded and we have a free trajectory again,
                # reset the attempt counter so the next stuck episode starts fresh.
                if self._escape_phase == 0 and self._selector_halt_streak == 0:
                    self._escape_attempt_count = 0
                traj = sel_result.trajectory
                ctx.waypoints = None  # recomputed below

        # --- Convert camera-frame trajectory → base_link waypoints → (v, w) ---
        if ctx.waypoints is None:
            ctx.waypoints = self._traj_ctrl.trajectory_to_waypoints(traj)
            ctx.v, ctx.w = self._traj_ctrl.fallback_proportional(ctx.waypoints)

    def _tick_escape_override(self, ctx: "_TickContext") -> bool:
        """Apply LiDAR escape and odom-based stuck detection; may override ctx.v/w.

        Also handles pose-goal proximity (language goals use the SM instead).
        Returns True if the tick should exit early (pose goal reached).
        """
        nb = _navdp_bridge
        odom = ctx.odom

        # --- Check goal proximity (pose goals only; language goals use the SM) ---
        if self._goal_pose is not None:
            gx = self._goal_pose.position.x
            gy = self._goal_pose.position.y
            dist = math.hypot(gx - odom[0], gy - odom[1])
            if dist < 0.5:
                self._goal_reached = True
                self._nav_state = NavigationState.IDLE
                self.cmd_vel.publish(Twist())
                logger.info("NavDP goal reached (dist=%.2f)", dist)
                return True

        # --- Escape override (LiDAR-based, when available) ---
        if self._escape_state is not None and self._latest_scan_points is not None:
            esc_result = nb["escape_tick"](
                state=self._escape_state,
                scan_points=self._latest_scan_points,
                odom_pose=odom,
                navdp_has_viable=True,
            )
            if esc_result.get("action") == "escape":
                goal_base = esc_result.get("goal_base", (0.0, 0.0))
                angle = math.atan2(goal_base[1], goal_base[0])
                ctx.v = 0.15 if abs(angle) < 0.5 else 0.0
                ctx.w = max(-0.5, min(0.5, angle))

        # --- Two-phase escape: backup then rotate toward frontier ---
        now = time.time()
        selector_stuck = self._selector_halt_streak >= 4

        if self._escape_phase == 1:
            # Phase 1: backing up
            elapsed = now - self._escape_phase_start
            if elapsed < self._escape_backup_duration:
                ctx.v = self._escape_backup_speed  # negative = reverse
                ctx.w = 0.0
            else:
                # Transition to phase 2: rotate toward frontier
                self._escape_phase = 2
                self._escape_phase_start = now

                # Determine rotation direction from cached frontier direction.
                # frontier_dir is (dx, dy) world-frame unit vector toward
                # the nearest frontier.  Convert to a target yaw and compute
                # the signed angular difference from the robot's current yaw.
                # On repeated attempts, override with larger forced rotation angles
                # in alternating directions to prevent getting locked in same spot.
                fdir = self._escape_frontier_dir
                attempt = self._escape_attempt_count
                if attempt >= 3:
                    # Escalate: force a 180° rotation, alternating left/right
                    forced_angle = math.pi if (attempt % 2 == 0) else -math.pi 
                    self._escape_rotate_w = math.copysign(0.5, forced_angle)
                    # Extend rotation duration proportionally
                    self._escape_rotate_duration = abs(forced_angle) / 0.5
                elif attempt >= 1:
                    # Second attempt: force 90° rotation away from wall,
                    # alternating direction each time
                    forced_angle = (math.pi / 2) * (1 if attempt % 2 == 0 else -1)
                    self._escape_rotate_w = math.copysign(0.5, forced_angle)
                    self._escape_rotate_duration = abs(forced_angle) / 0.5
                elif fdir is not None:
                    target_yaw = math.atan2(fdir[1], fdir[0])
                    _, _, cur_yaw = odom
                    # Signed shortest-arc difference
                    diff = (target_yaw - cur_yaw + math.pi) % (2 * math.pi) - math.pi
                    self._escape_rotate_w = max(-0.5, min(0.5, diff / self._escape_rotate_duration))
                else:
                    self._escape_rotate_w = 0.5  # default: turn left

                ctx.v = 0.0
                ctx.w = self._escape_rotate_w
                print(
                    f"[NavDP] escape phase 2: rotating w={self._escape_rotate_w:.2f} "
                    f"for {self._escape_rotate_duration:.1f}s "
                    f"(attempt={attempt}, frontier={'yes' if fdir is not None else 'no'})",
                    flush=True,
                )

        elif self._escape_phase == 2:
            # Phase 2: rotating toward frontier
            elapsed = now - self._escape_phase_start
            if elapsed < self._escape_rotate_duration:
                ctx.v = 0.0
                ctx.w = self._escape_rotate_w
            else:
                # Escape complete — resume normal policy
                self._escape_phase = 0
                self._odom_history.clear()
                self._selector_halt_streak = 0
                # Reset rotation duration to default for next escape
                self._escape_rotate_duration = 2.0
                print(
                    f"[NavDP] escape complete (backup+rotate attempt #{self._escape_attempt_count}), resuming policy",
                    flush=True,
                )

        elif selector_stuck or self._check_stuck(odom):
            # Trigger escape: start phase 1 (backup)
            self._escape_phase = 1
            self._escape_phase_start = now
            self._escape_attempt_count += 1
            trigger_reason = "selector_halt_streak" if selector_stuck else "odom_progress"

            # If stuck repeatedly, the guidance path likely leads into a dead-end — discard it
            if self._escape_attempt_count >= 2 and self._guidance_path is not None:
                with self._lock:
                    self._guidance_path = None
                    self._guidance_path_idx = 0
                print("[NavDP] Cleared stale guidance path after 2+ escape attempts", flush=True)

            # Cache the frontier direction now so it doesn't change mid-maneuver
            with self._lock:
                costmap = self._latest_costmap
            self._escape_frontier_dir = self._get_nearest_frontier_direction(odom, costmap)

            ctx.v = self._escape_backup_speed
            ctx.w = 0.0
            print(
                f"[NavDP] stuck at ({odom[0]:.2f}, {odom[1]:.2f}) via {trigger_reason} "
                f"— escape phase 1: backup {abs(self._escape_backup_speed):.2f}m/s "
                f"for {self._escape_backup_duration:.1f}s "
                f"(attempt #{self._escape_attempt_count}, "
                f"streak={self._selector_halt_streak}, "
                f"frontier={'yes' if self._escape_frontier_dir is not None else 'no'})",
                flush=True,
            )

        return False

    def _tick_publish(self, ctx: "_TickContext") -> None:
        """Publish cmd_vel and navdp_path; track last-good velocity."""
        # Log the very first successful cmd_vel for diagnostics
        if not getattr(self, "_cmd_vel_logged", False):
            self._cmd_vel_logged = True
            print(
                f"[NavDP] FIRST cmd_vel: v={ctx.v:.4f}, w={ctx.w:.4f}, "
                f"waypoints={ctx.waypoints.shape if ctx.waypoints is not None else None}, "
                f"first_wp="
                f"{ctx.waypoints[0].tolist() if ctx.waypoints is not None and len(ctx.waypoints) > 0 else None}, "
                f"traj[0]="
                f"{ctx.traj[0, :3].tolist() if ctx.traj is not None and ctx.traj.ndim >= 2 else None}, "
                f"cmd_vel transport={getattr(self.cmd_vel, '_transport', 'MISSING')}",
                flush=True,
            )

        twist = Twist(
            linear=Vector3(x=float(ctx.v), y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=float(ctx.w)),
        )

        # --- Diagnostic: catch the case where selector requested a halt
        # (fallback_used=True) but cmd_vel is still non-zero.  That would
        # mean hold/decel or escape override re-introduced velocity; the
        # caller should see it in logs to track down the wall-drift bug. ---
        _fb = getattr(self, "_last_sel_fallback", False)
        _nonzero = abs(float(twist.linear.x)) > 1e-3 or abs(float(twist.angular.z)) > 1e-3
        if _fb and _nonzero:
            # Unthrottled — this is the exact "walked into wall" signature.
            print(
                f"[NavDP-DIAG] selector halted but cmd_vel non-zero: "
                f"v={twist.linear.x:.3f} w={twist.angular.z:.3f} "
                f"sel_idx={getattr(self, '_last_sel_index', -1)} "
                f"sel_cost={getattr(self, '_last_sel_cost', 0.0):.1f} "
                f"escape_phase={self._escape_phase} "
                f"halt_streak={self._selector_halt_streak}",
                flush=True,
            )
        elif not hasattr(self, "_cmd_vel_diag_tick"):
            self._cmd_vel_diag_tick = 0
        else:
            self._cmd_vel_diag_tick += 1
            if self._cmd_vel_diag_tick % 30 == 1:  # ~every 5s at 6Hz
                _sm_state_str = "?"
                try:
                    _sm_state_str = self._state_machine.state.name
                except Exception:
                    pass
                _trail_len = len(self._exploration_trail) if hasattr(self, "_exploration_trail") else 0
                _guidance_wps = len(self._guidance_path) if getattr(self, "_guidance_path", None) is not None else 0
                print(
                    f"[NavDP-DIAG] cmd_vel v={twist.linear.x:.3f} w={twist.angular.z:.3f} "
                    f"nav={self._nav_state.name} sm={_sm_state_str} "
                    f"goal='{self._language_goal[:30]}' infer={self._infer_mode} "
                    f"skip_vlm={self._skip_vlm_detection} "
                    f"sel_fallback={_fb} sel_idx={getattr(self, '_last_sel_index', -1)} "
                    f"sel_cost={getattr(self, '_last_sel_cost', 0.0):.1f} "
                    f"escape_phase={self._escape_phase} "
                    f"halt_streak={self._selector_halt_streak} "
                    f"trail_len={_trail_len} guidance_wps={_guidance_wps}",
                    flush=True,
                )

        self.cmd_vel.publish(twist)
        # Track last good velocity for graceful network-failure hold/decel
        self._last_good_v = float(twist.linear.x)
        self._last_good_w = float(twist.angular.z)
        self._last_good_time = time.time()

        # --- Publish navdp_path for WebSocket 2D visualizer (no Rerun overhead) ---
        try:
            if ctx.waypoints is not None and len(ctx.waypoints) > 0:
                ox, oy, oyaw = ctx.odom
                cos_yaw = math.cos(oyaw)
                sin_yaw = math.sin(oyaw)
                poses = []
                for wp in ctx.waypoints:
                    wx = ox + cos_yaw * wp[0] - sin_yaw * wp[1]
                    wy = oy + sin_yaw * wp[0] + cos_yaw * wp[1]
                    poses.append(PoseStamped(position=[wx, wy, 0.0]))
                self.navdp_path.publish(Path(poses=poses, frame_id="world"))
        except Exception as e:
            if not getattr(self, "_path_pub_err_logged", False):
                self._path_pub_err_logged = True
                print(f"[NavDP] navdp_path publish error: {e}", flush=True)

    # ------------------------------------------------------------------
    # _tick orchestrator
    # ------------------------------------------------------------------

    def _tick(self) -> None:
        """Single control tick using the NavDP state machine.

        State machine flow:
            IDLE → (set_language_goal) → SEEK → (VLM found) → APPROACH → (depth < thresh) → STOPPED
                                          ↑                      │
                                          └──────────────────────┘ (lost target)
        """
        ctx = self._tick_guard_checks()
        if ctx is None:
            return

        self._tick_vlm_detection(ctx)

        if self._tick_state_machine_update(ctx):
            return  # goal reached → STOPPED

        if not self._tick_read_trajectory(ctx):
            return  # stale/missing trajectory — hold/decel published

        self._tick_select_trajectory(ctx)
        if self._tick_escape_override(ctx):
            return  # pose goal reached

        self._tick_publish(ctx)

    # --- LiDAR injection (called by NavDPMemory or external module) ---

    @rpc
    def set_scan_points(self, points: list[list[float]]) -> None:
        """Inject LiDAR scan points for escape controller.

        Args:
            points: list of [x, y] points in base_link frame.
        """
        with self._lock:
            self._latest_scan_points = np.array(points, dtype=np.float32)

    # --- Costmap injection for trajectory selector ---

    @rpc
    def set_costmap(self, costmap: OccupancyGrid) -> None:
        """Inject an occupancy grid for the trajectory selector.

        Args:
            costmap: OccupancyGrid with cost values (0=free, 100=lethal).
        """
        with self._lock:
            self._latest_costmap = costmap

    # --- Trajectory selector controls ---

    @rpc
    def enable_trajectory_selector(self) -> None:
        """Enable costmap-based trajectory selection at runtime."""
        self._trajectory_selector.enable()

    @rpc
    def disable_trajectory_selector(self) -> None:
        """Disable costmap-based trajectory selection at runtime."""
        self._trajectory_selector.disable()


navdp_navigator = NavDPNavigator.blueprint

__all__ = ["NavDPNavigator", "navdp_navigator"]

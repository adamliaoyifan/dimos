"""NavDP Navigator — DimOS Module wrapping the diffusion-policy state machine.

Implements ``NavigationInterface`` so DimOS agents / skill containers can
call ``set_goal()`` / ``cancel_goal()`` exactly like the built-in A* planner.

Internal pipeline (per tick @ ~10 Hz):
    Image + Odom ─► DiffusionPolicy(HTTP) ─► trajectory ─► TrajectoryController
                                                             ─► cmd_vel (Twist)
    LiDAR ─► EscapeController (if stuck) ─► override cmd_vel

State machine (mirroring navdp_bridge_node):
    IDLE → SEEK → APPROACH → STOPPED
              ↑       │
              └───────┘ (lost target)

    SEEK    = nogoal_step  (frontier exploration via diffusion policy)
    APPROACH = imagegoal_step (drive toward captured reference image)
    STOPPED  = zero velocity (depth < threshold)

Requires the NavDP inference server and VLM server running externally.
"""

from __future__ import annotations

import logging
import math
import sys
import threading
import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.nav_msgs.Path import Path
from dimos.msgs.sensor_msgs.Image import Image
from dimos.navigation.base import NavigationInterface, NavigationState

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
    In[Image]        color_image   — camera feed from robot
    In[PoseStamped]  odom          — robot odometry
    Out[Twist]       cmd_vel       — velocity commands to robot

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
    """

    # --- DimOS streams ---
    color_image: In[Image]
    odom: In[PoseStamped]
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

        # Runtime state (set in start())
        self._state_machine = None
        self._goal_context = None
        self._navdp_client = None
        self._vlm_client = None
        self._traj_ctrl = None
        self._escape_state = None

        # Latest sensor data (protected by lock)
        self._lock = threading.Lock()
        self._latest_image: np.ndarray | None = None
        self._latest_odom: tuple[float, float, float] | None = None
        self._latest_scan_points: np.ndarray | None = None

        # Navigation state
        self._nav_state = NavigationState.IDLE
        self._goal_pose: PoseStamped | None = None
        self._goal_reached = False
        self._running = False
        self._tick_thread: threading.Thread | None = None

        # Language goal for VLM-driven navigation
        self._language_goal: str = ""

        # VLM detection tracking
        self._last_vlm_check: float = 0.0
        self._last_vlm_mode: str = "unknown"
        self._last_vlm_conf: float = 0.0

        # Visualization rate limiting (avoid flooding Rerun)
        self._last_vis_time: float = 0.0
        self._vis_interval: float = 0.2  # 5 Hz max for debug overlay

        super().__init__(**kwargs)

    @rpc
    def start(self) -> None:
        super().start()
        try:
            _ensure_navdp_imports()
        except ImportError:
            logger.error(
                "NavDPNavigator disabled — navdp_bridge not installed. "
                "Add NavDP's ros2_ws/src/navdp_bridge to PYTHONPATH."
            )
            return
        nb = _navdp_bridge

        # Initialize NavDP components
        intrinsic = self._cam_intrinsic
        if intrinsic is None:
            intrinsic = np.array(
                [[460, 0, 320], [0, 460, 240], [0, 0, 1]], dtype=np.float32
            )
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
        self._disposables.add(
            Disposable(self.color_image.subscribe(self._on_image))
        )
        self._disposables.add(
            Disposable(self.odom.subscribe(self._on_odom))
        )

        # Start control loop
        self._running = True
        self._tick_thread = threading.Thread(
            target=self._control_loop, daemon=True, name="navdp-control"
        )
        self._tick_thread.start()
        logger.info("NavDPNavigator started (server=%s)", self._navdp_url)

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._tick_thread is not None:
            self._tick_thread.join(timeout=3.0)
        # Publish zero velocity
        self.cmd_vel.publish(Twist())
        super().stop()

    # --- Stream callbacks ---

    def _on_image(self, img: Image) -> None:
        with self._lock:
            self._latest_image = _image_to_bgr(img)

    def _on_odom(self, odom: PoseStamped) -> None:
        with self._lock:
            self._latest_odom = _pose_to_xyyaw(odom)

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
        with self._lock:
            self._goal_pose = None
            self._language_goal = ""
            self._nav_state = NavigationState.IDLE
            self._goal_reached = False
        if self._state_machine is not None:
            self._state_machine.clear_goal()
        if self._goal_context is not None:
            self._goal_context.clear_language_goal()
        self.cmd_vel.publish(Twist())
        logger.info("NavDP goal cancelled")
        return True

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
        with self._lock:
            self._language_goal = goal
            self._goal_reached = False
            self._nav_state = NavigationState.FOLLOWING_PATH
            self._last_vlm_mode = "unknown"
            self._last_vlm_conf = 0.0
        if self._goal_context is not None:
            self._goal_context.set_language_goal(goal)
        if self._state_machine is not None:
            self._state_machine.set_goal(has_memory_clue=False)
        logger.info("NavDP language goal: %r (state → SEEK)", goal)
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
        self._goal_context.set_reference_from_transition(image_bgr)
        # Force state machine to APPROACH
        self._state_machine.state = nb["NavState"].APPROACH
        self._state_machine.lost_count = 0
        logger.info("NavDP reference image set externally → APPROACH")
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
        except Exception:
            logger.debug("VLM detection query failed", exc_info=True)

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

    def _control_loop(self) -> None:
        """Main tick loop — runs diffusion policy and publishes cmd_vel."""
        dt = 1.0 / self._tick_rate_hz
        while self._running:
            try:
                self._tick()
            except Exception:
                logger.exception("NavDP control tick error")
            time.sleep(dt)

    def _tick(self) -> None:
        """Single control tick using the NavDP state machine.

        State machine flow:
            IDLE → (set_language_goal) → SEEK → (VLM found) → APPROACH → (depth < thresh) → STOPPED
                                          ↑                      │
                                          └──────────────────────┘ (lost target)
        """
        with self._lock:
            image = self._latest_image
            odom = self._latest_odom

        if image is None or odom is None:
            return  # waiting for sensor data

        if self._nav_state == NavigationState.IDLE:
            return  # no goal active

        nb = _navdp_bridge
        if nb is None:
            return

        NavState = nb["NavState"]

        # --- Run VLM detection periodically (for state machine transitions) ---
        if self._language_goal and self._state_machine is not None:
            sm_state = self._state_machine.state
            if sm_state in (NavState.SEEK, NavState.APPROACH):
                self._run_vlm_detection(image)

        # --- Update state machine ---
        # nogoal_step requires (rgb, depth); use a zero depth plane when no
        # depth sensor is available (NavDP uses it only for 3-D point lifting).
        depth = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        depth_ahead = nb["get_depth_ahead"](depth)

        old_state = None
        if self._state_machine is not None:
            old_state = self._state_machine.state
            self._state_machine.update(
                vlm_mode=self._last_vlm_mode,
                vlm_conf=self._last_vlm_conf,
                depth_ahead=depth_ahead,
            )
            new_state = self._state_machine.state

            # Handle SEEK → APPROACH transition: capture reference image
            if old_state == NavState.SEEK and new_state == NavState.APPROACH:
                ref_img, image_src = self._goal_context.set_reference_from_transition(image)
                logger.info(
                    "NavDP: target detected (conf=%.2f), using %s → APPROACH",
                    self._last_vlm_conf, image_src,
                )

            # Handle APPROACH → STOPPED: goal reached
            if old_state == NavState.APPROACH and new_state == NavState.STOPPED:
                logger.info("NavDP: goal reached (depth=%.2f) → STOPPED", depth_ahead)
                self._goal_reached = True
                self._nav_state = NavigationState.IDLE
                self._state_machine.reset()
                self.cmd_vel.publish(Twist())
                return

            # Handle APPROACH → SEEK: lost target
            if old_state == NavState.APPROACH and new_state == NavState.SEEK:
                logger.info("NavDP: target lost in APPROACH → SEEK (nogoal)")

        # --- Dispatch NavDP call based on state ---
        sm_state = self._state_machine.state if self._state_machine else NavState.SEEK
        ref_img = self._goal_context.get_reference_image() if self._goal_context else None

        traj = None
        all_traj = None
        all_vals = None
        if sm_state == NavState.APPROACH and ref_img is not None:
            # APPROACH mode: use imagegoal_step with reference image
            try:
                traj, all_traj, all_vals = self._navdp_client.imagegoal_step(
                    image, depth, ref_img
                )
            except Exception:
                logger.warning("NavDP imagegoal inference failed")
                traj = None
        else:
            # IDLE/SEEK/NAVIGATE: use nogoal_step for exploration
            try:
                traj, all_traj, all_vals = self._navdp_client.nogoal_step(
                    image, depth
                )
            except Exception:
                logger.warning("NavDP nogoal inference failed, sending zero cmd_vel")
                self.cmd_vel.publish(Twist())
                return

        if traj is None:
            return

        # --- Convert camera-frame trajectory → base_link waypoints → velocity ---
        waypoints = self._traj_ctrl.trajectory_to_waypoints(traj)
        v, w = self._traj_ctrl.fallback_proportional(waypoints)

        # --- Check goal proximity (for pose goals, not language goals) ---
        if self._goal_pose is not None:
            gx = self._goal_pose.position.x
            gy = self._goal_pose.position.y
            dist = math.hypot(gx - odom[0], gy - odom[1])
            if dist < 0.5:
                self._goal_reached = True
                self._nav_state = NavigationState.IDLE
                self.cmd_vel.publish(Twist())
                logger.info("NavDP goal reached (dist=%.2f)", dist)
                return

        # --- Escape override ---
        if self._escape_state is not None and self._latest_scan_points is not None:
            esc_result = nb["escape_tick"](
                state=self._escape_state,
                scan_points=self._latest_scan_points,
                odom_pose=odom,
                navdp_has_viable=True,
            )
            if esc_result.get("action") == "escape":
                # Override with escape velocity
                goal_base = esc_result.get("goal_base", (0.0, 0.0))
                angle = math.atan2(goal_base[1], goal_base[0])
                v = 0.15 if abs(angle) < 0.5 else 0.0
                w = max(-0.5, min(0.5, angle))

        # --- Debug visualization (Rerun 3D + navdp_path stream) ---
        self._publish_debug_vis(
            all_traj=all_traj,
            all_vals=all_vals,
            waypoints=waypoints,
            odom=odom,
            v=v,
            w=w,
        )

        # --- Publish ---
        twist = Twist(
            linear=Vector3(x=float(v), y=0.0, z=0.0),
            angular=Vector3(x=0.0, y=0.0, z=float(w)),
        )
        self.cmd_vel.publish(twist)

    # --- LiDAR injection (called by NavDPMemory or external module) ---

    @rpc
    def set_scan_points(self, points: list[list[float]]) -> None:
        """Inject LiDAR scan points for escape controller.

        Args:
            points: list of [x, y] points in base_link frame.
        """
        with self._lock:
            self._latest_scan_points = np.array(points, dtype=np.float32)


navdp_navigator = NavDPNavigator.blueprint

__all__ = ["NavDPNavigator", "navdp_navigator"]

"""NavDP Navigator — DimOS Module wrapping the diffusion-policy state machine.

Implements ``NavigationInterface`` so DimOS agents / skill containers can
call ``set_goal()`` / ``cancel_goal()`` exactly like the built-in A* planner.

Internal pipeline (per tick @ ~10 Hz):
    Image + Odom ─► DiffusionPolicy(HTTP) ─► trajectory ─► TrajectoryController
                                                             ─► cmd_vel (Twist)
    LiDAR ─► EscapeController (if stuck) ─► override cmd_vel

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
from dimos.msgs.geometry_msgs import PoseStamped, Twist, Vector3
from dimos.msgs.sensor_msgs import Image
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
    from dimos.msgs.sensor_msgs import ImageFormat

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

    def __init__(
        self,
        navdp_server_url: str = "http://127.0.0.1:8880",
        vlm_server_url: str = "http://127.0.0.1:8866",
        goal_image_dir: str = "/tmp/navdp_goals",
        tick_rate_hz: float = 10.0,
        cam_intrinsic: np.ndarray | None = None,
    ) -> None:
        self._navdp_url = navdp_server_url
        self._vlm_url = vlm_server_url
        self._goal_image_dir = goal_image_dir
        self._tick_rate_hz = tick_rate_hz
        self._cam_intrinsic = cam_intrinsic

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

        super().__init__()

    @rpc
    def start(self) -> None:
        super().start()
        _ensure_navdp_imports()
        nb = _navdp_bridge

        # Initialize NavDP components
        intrinsic = self._cam_intrinsic
        if intrinsic is None:
            intrinsic = np.array(
                [[460, 0, 320], [0, 460, 240], [0, 0, 1]], dtype=np.float32
            )

        self._state_machine = nb["NavigationStateMachine"]()
        self._goal_context = nb["GoalContext"](self._goal_image_dir)
        self._navdp_client = nb["NavDPClient"](
            server_url=self._navdp_url,
            cam_intrinsic=intrinsic,
        )
        self._vlm_client = nb["VLMClient"](
            vlm_url=self._vlm_url,
            logger=logger,
        )
        self._traj_ctrl = nb["TrajectoryController"]()
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
        self.cmd_vel.publish(Twist())
        logger.info("NavDP goal cancelled")
        return True

    # --- NavDP-specific RPCs ---

    @rpc
    def set_language_goal(self, goal: str) -> bool:
        """Set a language-based navigation goal (e.g., 'find the glasses on desk').

        This triggers VLM-driven hierarchical search via the NavDP state machine.

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
        if self._goal_context is not None:
            self._goal_context.set_language_goal(goal)
        logger.info("NavDP language goal: %r", goal)
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
        """Single control tick."""
        with self._lock:
            image = self._latest_image
            odom = self._latest_odom

        if image is None or odom is None:
            return  # waiting for sensor data

        if self._nav_state == NavigationState.IDLE:
            return  # no goal active

        # --- Run diffusion policy ---
        try:
            traj, _, _ = self._navdp_client.nogoal_step(image)
        except Exception:
            logger.warning("NavDP inference failed, sending zero cmd_vel")
            self.cmd_vel.publish(Twist())
            return

        if traj is None:
            return

        # Squeeze batch dim if present
        if traj.ndim == 3:
            traj = traj[0]

        # --- Trajectory → velocity ---
        v, w = self._traj_ctrl.fallback_proportional(traj)

        # --- Check goal proximity ---
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
            nb = _navdp_bridge
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

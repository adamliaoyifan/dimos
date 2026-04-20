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

"""Frontier overlay — projects frontier centroids onto camera images.

Subscribes:
    - color_image (pSHM)   — live camera frame
    - odom (pSHM)          — robot pose in world frame

Polls:
    - WavefrontFrontierExplorer.get_frontier_points() via RPC

Publishes:
    - frontier_overlay_image (pSHM) — camera image with frontier markers drawn

Coordinate transform chain:
    frontier (world x, y, z=0)
    → base_link (subtract odom, inverse quaternion rotation)
    → camera optical frame (inverse cam-to-base rotation, subtract translation)
    → pixel (u, v) via pinhole model K @ [X/Z, Y/Z, 1]

The transform is the **inverse** of ObjectLocalizer's back-projection:
    ObjectLocalizer: pixel → camera → base_link → world
    FrontierOverlay: world → base_link → camera → pixel
"""

from __future__ import annotations

import logging
import math
import threading
import time
from typing import Any

import cv2
import numpy as np

from dimos.core.module import Module, ModuleConfig, rpc
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from reactivex.disposable import Disposable

logger = logging.getLogger(__name__)


class FrontierOverlayConfig(ModuleConfig):
    """Configuration for FrontierOverlayModule."""

    poll_interval_s: float = 0.5
    """How often to poll WavefrontFrontierExplorer for frontier points (seconds)."""

    cam_x: float = 0.13
    cam_y: float = 0.00
    cam_z: float = 0.30
    cam_pitch: float = 0.157

    cam_intrinsic: list[list[float]] | None = None
    """3x3 camera intrinsic matrix [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]."""

    marker_radius: int = 10
    """Pixel radius of frontier markers."""

    selected_color: tuple[int, int, int] = (0, 255, 0)
    """BGR color for the selected frontier goal."""

    candidate_color: tuple[int, int, int] = (0, 200, 255)
    """BGR color for unselected frontier candidates."""

    label_font_scale: float = 0.5
    """Font scale for distance labels."""


class FrontierOverlayModule(Module[FrontierOverlayConfig]):
    """Projects frontier centroids onto the camera image and publishes an overlay.

    This module is the frontier equivalent of NavDP trajectory visualization —
    it lets you see where frontiers are relative to the robot's camera view.
    """

    default_config = FrontierOverlayConfig

    rpc_calls: list[str] = [
        "WavefrontFrontierExplorer.get_frontier_points",
    ]

    color_image: In[Image]
    odom: In[PoseStamped]
    frontier_overlay_image: Out[Image]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._latest_image: Image | None = None
        self._latest_odom: PoseStamped | None = None
        self._frontiers: list[dict[str, float]] = []
        self._lock = threading.Lock()
        self._poll_thread: threading.Thread | None = None
        self._stop_event = threading.Event()

        # Build camera transforms once in start()
        self._K: np.ndarray | None = None
        self._R_base_to_cam: np.ndarray | None = None
        self._t_cam_in_base: np.ndarray | None = None

    @rpc
    def start(self) -> None:
        super().start()

        # Build intrinsic matrix
        if self.config.cam_intrinsic is not None:
            self._K = np.array(self.config.cam_intrinsic, dtype=np.float64)
        else:
            self._K = np.array(
                [[460, 0, 320], [0, 460, 240], [0, 0, 1]], dtype=np.float64
            )

        # Build extrinsic: camera → base_link (same convention as ObjectLocalizer)
        # Then invert for base_link → camera (forward projection).
        cos_p = math.cos(self.config.cam_pitch)
        sin_p = math.sin(self.config.cam_pitch)

        # Undo pitch in camera optical frame (rotation about camera Y)
        R_unpitch = np.array([
            [cos_p, 0.0, sin_p],
            [0.0, 1.0, 0.0],
            [-sin_p, 0.0, cos_p],
        ], dtype=np.float64)

        # Camera optical → base_link axis mapping
        # camera X → base -Y, camera Y → base -Z, camera Z → base +X
        R_opt_to_base = np.array([
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ], dtype=np.float64)

        R_cam_to_base = R_opt_to_base @ R_unpitch
        # Invert: base_link → camera optical
        self._R_base_to_cam = R_cam_to_base.T

        self._t_cam_in_base = np.array(
            [self.config.cam_x, self.config.cam_y, self.config.cam_z],
            dtype=np.float64,
        )

        # Subscribe streams
        unsub_img = self.color_image.subscribe(self._on_image)
        self._disposables.add(Disposable(unsub_img))
        unsub_odom = self.odom.subscribe(self._on_odom)
        self._disposables.add(Disposable(unsub_odom))

        # Start polling thread for frontier data
        self._stop_event.clear()
        self._poll_thread = threading.Thread(
            target=self._poll_frontiers, daemon=True, name="frontier_overlay_poll"
        )
        self._poll_thread.start()

        logger.info(
            "[FrontierOverlay] started — cam_pitch=%.3f, poll=%.1fs",
            self.config.cam_pitch, self.config.poll_interval_s,
        )

    @rpc
    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=3.0)
        super().stop()

    def _on_image(self, msg: Image) -> None:
        self._latest_image = msg

    def _on_odom(self, msg: PoseStamped) -> None:
        self._latest_odom = msg
        # Render overlay whenever we get new odom (ensures fresh data)
        self._render_overlay()

    def _poll_frontiers(self) -> None:
        """Poll WavefrontFrontierExplorer.get_frontier_points via RPC."""
        _logged_error = False
        while not self._stop_event.wait(self.config.poll_interval_s):
            try:
                rpc_call = self.get_rpc_calls(
                    "WavefrontFrontierExplorer.get_frontier_points"
                )
                result = rpc_call()
                with self._lock:
                    self._frontiers = result or []
                _logged_error = False
            except Exception as exc:
                if not _logged_error:
                    logger.warning("[FrontierOverlay] poll failed: %s", exc)
                    _logged_error = True

    # ------------------------------------------------------------------
    # World → pixel projection
    # ------------------------------------------------------------------

    def _world_to_pixel(
        self, wx: float, wy: float, wz: float = 0.0
    ) -> tuple[int, int] | None:
        """Project a world-frame 3D point to pixel coordinates.

        Returns None if the point is behind the camera.

        Transform chain:
            1. world → base_link: inverse of odom (R_odom.T @ (p_world - t_odom))
            2. base_link → camera: R_base_to_cam @ (p_base - t_cam_in_base)
            3. camera → pixel: K @ [X/Z, Y/Z, 1]
        """
        odom = self._latest_odom
        if odom is None or self._K is None or self._R_base_to_cam is None:
            return None

        # 1. World → base_link
        p_world = np.array([wx, wy, wz], dtype=np.float64)
        t_odom = np.array(
            [odom.position.x, odom.position.y, odom.position.z],
            dtype=np.float64,
        )
        q = odom.orientation
        R_odom = self._quat_to_rotation(q.x, q.y, q.z, q.w)
        p_base = R_odom.T @ (p_world - t_odom)

        # 2. Base_link → camera optical
        p_cam = self._R_base_to_cam @ (p_base - self._t_cam_in_base)

        # Behind camera? (camera +Z is forward)
        if p_cam[2] <= 0.05:
            return None

        # 3. Pinhole projection
        fx, fy = self._K[0, 0], self._K[1, 1]
        cx, cy = self._K[0, 2], self._K[1, 2]
        u = int(p_cam[0] * fx / p_cam[2] + cx)
        v = int(p_cam[1] * fy / p_cam[2] + cy)
        return (u, v)

    @staticmethod
    def _quat_to_rotation(
        qx: float, qy: float, qz: float, qw: float
    ) -> np.ndarray:
        """Convert quaternion (x, y, z, w) to 3x3 rotation matrix."""
        r = np.zeros((3, 3), dtype=np.float64)
        r[0, 0] = 1 - 2 * (qy * qy + qz * qz)
        r[0, 1] = 2 * (qx * qy - qz * qw)
        r[0, 2] = 2 * (qx * qz + qy * qw)
        r[1, 0] = 2 * (qx * qy + qz * qw)
        r[1, 1] = 1 - 2 * (qx * qx + qz * qz)
        r[1, 2] = 2 * (qy * qz - qx * qw)
        r[2, 0] = 2 * (qx * qz - qy * qw)
        r[2, 1] = 2 * (qy * qz + qx * qw)
        r[2, 2] = 1 - 2 * (qx * qx + qy * qy)
        return r

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def _render_overlay(self) -> None:
        """Draw frontier markers on the latest camera image and publish."""
        img = self._latest_image
        odom = self._latest_odom
        if img is None or odom is None or img.data is None:
            return

        with self._lock:
            frontiers = list(self._frontiers)

        # Copy image (RGB → BGR for OpenCV drawing)
        frame = img.data.copy()
        if frame.ndim == 3 and frame.shape[2] == 3:
            canvas = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        else:
            canvas = frame

        img_h, img_w = canvas.shape[:2]

        robot_x = odom.position.x
        robot_y = odom.position.y

        for i, fp in enumerate(frontiers):
            fx, fy = fp["x"], fp["y"]
            is_selected = fp.get("selected", 0.0) > 0.5

            pixel = self._world_to_pixel(fx, fy, 0.0)
            if pixel is None:
                continue
            u, v = pixel

            # Clip to image bounds (with margin for labels)
            if u < -50 or u > img_w + 50 or v < -50 or v > img_h + 50:
                continue

            # Clamp for drawing
            u_draw = max(0, min(u, img_w - 1))
            v_draw = max(0, min(v, img_h - 1))

            # Distance from robot
            dist = math.sqrt((fx - robot_x) ** 2 + (fy - robot_y) ** 2)

            # Draw marker
            color = self.config.selected_color if is_selected else self.config.candidate_color
            radius = self.config.marker_radius + (4 if is_selected else 0)
            thickness = -1 if is_selected else 2  # filled for selected

            cv2.circle(canvas, (u_draw, v_draw), radius, color, thickness)

            # Rank label + distance
            label = f"#{i + 1} {dist:.1f}m"
            if is_selected:
                label = f"GOAL {dist:.1f}m"
            cv2.putText(
                canvas, label,
                (u_draw + radius + 4, v_draw + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.config.label_font_scale,
                color, 1, cv2.LINE_AA,
            )

        # Convert back to RGB for DimOS Image
        if canvas.ndim == 3 and canvas.shape[2] == 3:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

        overlay_msg = Image(data=canvas, encoding="rgb8")
        self.frontier_overlay_image.publish(overlay_msg)

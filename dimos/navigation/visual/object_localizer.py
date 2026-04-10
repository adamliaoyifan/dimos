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

"""3D object localization from 2D bounding boxes and depth images.

Provides:
  - :class:`ObjectEstimate` — a single 3D position estimate for a detected object.
  - :class:`ObjectInstance` — instance-level tracking record with EMA-filtered position.
  - :class:`ObjectLocalizer` — back-projects a VLM bbox + depth ROI into world frame.

Back-projection math
--------------------
Given a bounding box (x1,y1,x2,y2) and an aligned depth image:

1. Compute ROI depth: extract the centre 60 % of the bbox to avoid depth edges.
2. Filter invalid pixels (0, NaN, inf) and outliers (> 2 × median).
3. Robust depth ``d`` = median of remaining valid pixels.
4. Back-project bbox centre pixel ``(cx_px, cy_px)`` to camera frame::

       p_cam = K^-1 @ [cx_px, cy_px, 1]^T * d
               (units: metres)

   Camera convention: +Z forward, +X right, +Y down.

5. Transform camera → base_link using extrinsics ``(cam_x, cam_y, cam_z, cam_pitch)``::

       p_base = R_cam_to_base @ p_cam + t_cam_in_base

   where ``R_cam_to_base`` accounts for the pitch tilt (positive = downward) and
   the axis-mapping between optical (+Z fwd) and base_link (+X fwd).

6. Transform base_link → world using the robot's odometry quaternion + position::

       p_world = R_odom @ p_base + t_odom
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from dimos.models.qwen.bbox import BBox
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.sensor_msgs.Image import Image
from dimos.utils.logging_config import setup_logger

logger = setup_logger()


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class ObjectEstimate:
    """A single 3D position estimate for a detected object.

    All coordinates are in the world (map) frame.
    """

    x: float
    y: float
    z: float
    confidence: float
    """Fraction of valid depth pixels in the bbox ROI (0–1)."""
    depth_m: float
    """Median raw depth in the bbox ROI (metres)."""
    method: str = "depth"
    """Source of the estimate: "depth" or "lidar_depth"."""


@dataclass
class ObjectInstance:
    """Instance-level tracking record for a detected object.

    Maintains an EMA-filtered world-frame position and observation statistics.
    """

    instance_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = ""
    """Semantic label from the VLM query."""
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    """EMA-filtered world-frame position."""
    confidence: float = 0.0
    depth_m: float = 0.0
    observation_count: int = 0
    last_seen_ts: float = field(default_factory=time.time)
    first_seen_ts: float = field(default_factory=time.time)
    first_seen_odom: Optional[PoseStamped] = None

    def update(self, estimate: ObjectEstimate, alpha: float = 0.3) -> None:
        """Apply EMA update from a new ObjectEstimate.

        Args:
            estimate: New 3D estimate to incorporate.
            alpha: EMA learning rate (higher = more weight on new estimate).
        """
        if self.observation_count == 0:
            self.x = estimate.x
            self.y = estimate.y
            self.z = estimate.z
        else:
            self.x = alpha * estimate.x + (1.0 - alpha) * self.x
            self.y = alpha * estimate.y + (1.0 - alpha) * self.y
            self.z = alpha * estimate.z + (1.0 - alpha) * self.z
        self.confidence = estimate.confidence
        self.depth_m = estimate.depth_m
        self.observation_count += 1
        self.last_seen_ts = time.time()

    def to_estimate(self) -> ObjectEstimate:
        """Return the current EMA position as an ObjectEstimate."""
        return ObjectEstimate(
            x=self.x,
            y=self.y,
            z=self.z,
            confidence=self.confidence,
            depth_m=self.depth_m,
            method="ema_filtered",
        )


# ---------------------------------------------------------------------------
# ObjectLocalizer
# ---------------------------------------------------------------------------


class ObjectLocalizer:
    """Back-projects a VLM bounding box + depth image to a world-frame 3D position.

    Args:
        intrinsic: 3×3 camera intrinsic matrix K (numpy float32/64).
        cam_x: Camera x offset in base_link frame (forward, metres).
        cam_y: Camera y offset in base_link frame (left, metres).
        cam_z: Camera z offset in base_link frame (up, metres).
        cam_pitch: Camera downward tilt angle in radians (positive = down).
        depth_scale: Multiplier to convert raw depth pixel values to metres.
                     RealSense typically stores depth in mm (scale=0.001);
                     the simulation depth image is already in metres (scale=1.0).
    """

    # Rotation from camera optical frame (+Z fwd, +X right, +Y down)
    # to base_link (+X fwd, +Y left, +Z up) — pure axis permutation, no pitch yet.
    # camera X  → base_link -Y  (right → -left)
    # camera Y  → base_link -Z  (down  → -up)
    # camera Z  → base_link +X  (forward → forward)
    _R_OPT_TO_BASE = np.array([
        [0.0,  0.0,  1.0],   # base_link X = camera Z
        [-1.0, 0.0,  0.0],   # base_link Y = -camera X
        [0.0, -1.0,  0.0],   # base_link Z = -camera Y
    ], dtype=np.float64)

    def __init__(
        self,
        intrinsic: np.ndarray,
        cam_x: float = 0.13,
        cam_y: float = 0.00,
        cam_z: float = 0.30,
        cam_pitch: float = 0.157,
        depth_scale: float = 1.0,
    ) -> None:
        self._K = np.asarray(intrinsic, dtype=np.float64)
        self._K_inv = np.linalg.inv(self._K)
        self._cam_x = cam_x
        self._cam_y = cam_y
        self._cam_z = cam_z
        self._cam_pitch = cam_pitch
        self._depth_scale = depth_scale

        # Build rotation: optical → base_link, accounting for pitch tilt.
        # The camera is tilted downward by cam_pitch about the camera's lateral
        # axis (Y in optical frame, which maps to -base_link Y after _R_OPT_TO_BASE).
        # We build the compound rotation: first undo the pitch in camera space,
        # then apply the axis-mapping.
        cos_p = math.cos(cam_pitch)
        sin_p = math.sin(cam_pitch)
        # Rotation about camera Y axis by -pitch (tilt camera up to align with base)
        R_unpitch = np.array([
            [cos_p,  0.0, sin_p],
            [0.0,    1.0, 0.0  ],
            [-sin_p, 0.0, cos_p],
        ], dtype=np.float64)
        self._R_cam_to_base: np.ndarray = self._R_OPT_TO_BASE @ R_unpitch
        # Translation: camera origin in base_link frame
        self._t_cam_in_base = np.array([cam_x, cam_y, cam_z], dtype=np.float64)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def estimate_from_bbox_depth(
        self,
        bbox: BBox,
        depth_image: Image,
        odom: PoseStamped,
        roi_fraction: float = 0.6,
        outlier_factor: float = 2.0,
        min_confidence: float = 0.05,
        max_depth: float = 5.0,
    ) -> ObjectEstimate | None:
        """Estimate 3D world-frame position of an object from its bounding box and depth.

        Args:
            bbox: Bounding box as (x1, y1, x2, y2).  May be in pixel coordinates
                  or 0–1000 scale (auto-detected from depth_image shape).
            depth_image: Aligned depth image (single-channel or 3-channel where
                         channel 0 is depth).  Values should be in metres
                         (or raw counts if depth_scale != 1.0).
            odom: Current robot pose in world frame.
            roi_fraction: Inner fraction of the bbox to sample for depth.
                          E.g. 0.6 means centre 60 % in each dimension.
            outlier_factor: Reject pixels deeper than ``outlier_factor * median``.
            min_confidence: Minimum valid-pixel fraction to trust the estimate.
            max_depth: Clamp depth to this value (metres).  At long range,
                       small pixel errors in the bbox are amplified into large
                       lateral position errors.  Clamping prevents wildly wrong
                       3D estimates when the object is far away.

        Returns:
            :class:`ObjectEstimate` on success, or ``None`` if depth is
            unavailable or confidence is below ``min_confidence``.
        """
        depth_data = depth_image.data
        if depth_data is None or depth_data.size == 0:
            return None

        img_h, img_w = depth_data.shape[:2]

        # --- Normalise bbox to pixel coordinates ---
        x1, y1, x2, y2 = bbox
        if x2 > img_w or y2 > img_h:
            # 0–1000 scale → pixel
            x1 = x1 / 1000.0 * img_w
            y1 = y1 / 1000.0 * img_h
            x2 = x2 / 1000.0 * img_w
            y2 = y2 / 1000.0 * img_h

        x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)

        # --- Extract centre ROI to avoid depth edges ---
        bw = x2 - x1
        bh = y2 - y1
        margin_x = bw * (1.0 - roi_fraction) / 2.0
        margin_y = bh * (1.0 - roi_fraction) / 2.0
        rx1 = max(0, int(x1 + margin_x))
        ry1 = max(0, int(y1 + margin_y))
        rx2 = min(img_w - 1, int(x2 - margin_x))
        ry2 = min(img_h - 1, int(y2 - margin_y))

        if rx2 <= rx1 or ry2 <= ry1:
            # Bbox too small — fall back to full bbox
            rx1, ry1 = max(0, int(x1)), max(0, int(y1))
            rx2, ry2 = min(img_w - 1, int(x2)), min(img_h - 1, int(y2))

        # --- Extract depth ROI ---
        if depth_data.ndim == 3:
            roi = depth_data[ry1:ry2, rx1:rx2, 0].astype(np.float64)
        else:
            roi = depth_data[ry1:ry2, rx1:rx2].astype(np.float64)

        roi = roi * self._depth_scale

        # --- Filter invalid pixels ---
        valid_mask = np.isfinite(roi) & (roi > 0.0)
        total_pixels = roi.size
        valid_pixels = int(np.sum(valid_mask))

        if valid_pixels == 0:
            logger.debug("[ObjectLocalizer] No valid depth pixels in bbox ROI")
            return None

        valid_depths = roi[valid_mask]
        median_depth = float(np.median(valid_depths))

        # Reject outliers beyond outlier_factor × median
        inlier_mask = valid_depths <= outlier_factor * median_depth
        inlier_depths = valid_depths[inlier_mask]

        if inlier_depths.size == 0:
            return None

        robust_depth = float(np.median(inlier_depths))
        # Clamp depth to reduce lateral error amplification at long range.
        if robust_depth > max_depth:
            logger.debug(
                "[ObjectLocalizer] Clamping depth %.2fm → %.2fm",
                robust_depth, max_depth,
            )
            robust_depth = max_depth
        confidence = float(inlier_depths.size) / max(total_pixels, 1)

        if confidence < min_confidence:
            logger.debug(
                "[ObjectLocalizer] Low depth confidence %.2f (< %.2f)",
                confidence, min_confidence,
            )
            return None

        # --- Bbox centre pixel ---
        cx_px = (x1 + x2) / 2.0
        cy_px = (y1 + y2) / 2.0

        # --- Back-project to camera optical frame ---
        pixel_h = np.array([cx_px, cy_px, 1.0], dtype=np.float64)
        ray = self._K_inv @ pixel_h          # unit direction in camera frame
        p_cam = ray * robust_depth           # scale to depth

        # --- Transform camera → base_link ---
        p_base = self._R_cam_to_base @ p_cam + self._t_cam_in_base

        # --- Lateral sanity check ---
        # If the object appears more than ~35° off the robot's forward axis in
        # base_link frame the back-projection is too noisy to be trusted as a
        # 3D position estimate.  The approach code will fall back to bbox bearing.
        forward_m = p_base[0]   # base_link +X = forward
        lateral_m = p_base[1]   # base_link +Y = left
        if forward_m > 0.1 and abs(lateral_m / forward_m) > 0.7:
            logger.debug(
                "[ObjectLocalizer] Rejecting off-axis estimate (lat/fwd=%.2f)",
                lateral_m / forward_m,
            )
            return None

        # --- Transform base_link → world using odom ---
        p_world = self._base_to_world(p_base, odom)

        logger.debug(
            "[ObjectLocalizer] depth=%.2fm conf=%.2f p_cam=%s p_world=%s",
            robust_depth, confidence,
            [round(v, 3) for v in p_cam.tolist()],
            [round(v, 3) for v in p_world.tolist()],
        )

        return ObjectEstimate(
            x=float(p_world[0]),
            y=float(p_world[1]),
            z=float(p_world[2]),
            confidence=confidence,
            depth_m=robust_depth,
            method="depth",
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _base_to_world(p_base: np.ndarray, odom: PoseStamped) -> np.ndarray:
        """Transform a point from base_link to world frame using odometry.

        Args:
            p_base: 3-element array [x, y, z] in base_link frame.
            odom: Robot pose (position + orientation quaternion) in world frame.

        Returns:
            3-element array [x, y, z] in world frame.
        """
        q = odom.orientation
        # Build rotation matrix from quaternion (Hamilton product convention)
        qx, qy, qz, qw = q.x, q.y, q.z, q.w
        R = np.array([
            [1 - 2*(qy*qy + qz*qz),   2*(qx*qy - qz*qw),   2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw),   1 - 2*(qx*qx + qz*qz),   2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw),       2*(qy*qz + qx*qw), 1 - 2*(qx*qx + qy*qy)],
        ], dtype=np.float64)
        t = np.array([odom.position.x, odom.position.y, odom.position.z], dtype=np.float64)
        return R @ p_base + t

    @staticmethod
    def compute_approach_goal(
        odom: PoseStamped,
        target: ObjectEstimate,
        stop_distance: float = 0.6,
        max_step_distance: float = 1.0,
    ) -> tuple[float, float, float]:
        """Compute the A* goal position along the robot→target line.

        The goal is placed along the straight line from the robot to the
        target.  To stay within the mapped costmap, the goal is clamped to
        at most ``max_step_distance`` from the robot.  When the robot is
        close enough the goal stops at ``stop_distance`` before the target.

        Args:
            odom: Current robot pose.
            target: Target 3D world-frame estimate.
            stop_distance: Distance from the target to stop at (metres).
            max_step_distance: Maximum distance from the robot to place
                the goal in a single step (metres).  Prevents the A*
                planner from receiving goals outside the explored map.

        Returns:
            ``(goal_x, goal_y, goal_yaw)`` — 2D goal position and approach yaw.
        """
        rx, ry = odom.position.x, odom.position.y
        tx, ty = target.x, target.y
        dist = math.hypot(tx - rx, ty - ry)
        yaw = math.atan2(ty - ry, tx - rx)

        if dist <= stop_distance:
            return rx, ry, yaw

        # Desired advance: full remaining distance minus the stop buffer
        desired_advance = dist - stop_distance
        # Clamp to max_step_distance so the goal stays on the explored map
        step = min(desired_advance, max_step_distance)

        dx = tx - rx
        dy = ty - ry
        goal_x = rx + (dx / dist) * step
        goal_y = ry + (dy / dist) * step
        return goal_x, goal_y, yaw


__all__ = ["ObjectEstimate", "ObjectInstance", "ObjectLocalizer"]

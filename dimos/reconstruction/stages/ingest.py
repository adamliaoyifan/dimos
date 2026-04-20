# Copyright 2025-2026 Dimensional Inc.
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

"""Data ingestion: load images, LiDAR, camera poses into pipeline types."""

from __future__ import annotations

import json
import logging
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dimos.reconstruction.config import IngestConfig
from dimos.reconstruction.types import (
    CameraIntrinsics,
    CameraPose,
    ImageDataset,
    PointCloudData,
)

logger = logging.getLogger(__name__)


@dataclass
class IngestInput:
    """Marker input for the first pipeline stage."""

    pass


@dataclass
class IngestOutput:
    """Output of the ingestion stage."""

    image_dataset: ImageDataset | None = None
    lidar_point_cloud: PointCloudData | None = None


class IngestStage:
    """Load raw data from disk into pipeline data types."""

    def __init__(self, config: IngestConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "ingest"

    def run(self, input_data: IngestInput) -> IngestOutput:
        output = IngestOutput()

        if self.config.image_dir and self.config.pose_file:
            if self.config.pose_format == "nerfstudio":
                output.image_dataset = load_nerfstudio_poses(self.config.pose_file)
            elif self.config.pose_format == "colmap":
                output.image_dataset = load_colmap_poses(self.config.pose_file)
            else:
                raise ValueError(f"Unknown pose format: {self.config.pose_format}")

        if self.config.lidar_dir:
            output.lidar_point_cloud = load_lidar_point_cloud(
                self.config.lidar_dir,
                extensions=[".ply", ".pcd", ".bin", ".las"],
            )

        return output

    def validate_input(self, input_data: IngestInput) -> list[str]:
        errors: list[str] = []
        if self.config.image_dir and not self.config.image_dir.exists():
            errors.append(f"Image directory not found: {self.config.image_dir}")
        if self.config.lidar_dir and not self.config.lidar_dir.exists():
            errors.append(f"LiDAR directory not found: {self.config.lidar_dir}")
        if self.config.pose_file and not self.config.pose_file.exists():
            errors.append(f"Pose file not found: {self.config.pose_file}")
        return errors


# --- Pure functions ---


def load_colmap_poses(sparse_dir: Path) -> ImageDataset:
    """Parse COLMAP sparse reconstruction (binary or text) into ImageDataset.

    Expects images.bin / images.txt and cameras.bin / cameras.txt in sparse_dir.
    """
    images_bin = sparse_dir / "images.bin"
    cameras_bin = sparse_dir / "cameras.bin"

    if images_bin.exists() and cameras_bin.exists():
        return _load_colmap_binary(sparse_dir)

    images_txt = sparse_dir / "images.txt"
    cameras_txt = sparse_dir / "cameras.txt"
    if images_txt.exists() and cameras_txt.exists():
        return _load_colmap_text(sparse_dir)

    raise FileNotFoundError(
        f"No COLMAP model found in {sparse_dir}. "
        "Expected images.bin/cameras.bin or images.txt/cameras.txt"
    )


def _load_colmap_text(sparse_dir: Path) -> ImageDataset:
    """Load COLMAP text format."""
    # Parse cameras.txt
    intrinsics = _parse_colmap_cameras_txt(sparse_dir / "cameras.txt")

    # Parse images.txt
    images: list[CameraPose] = []
    with open(sparse_dir / "images.txt") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    # images.txt has 2 lines per image: metadata, then 2D points
    for i in range(0, len(lines), 2):
        parts = lines[i].split()
        # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
        qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
        image_name = parts[9]

        R = _quat_to_rot(qw, qx, qy, qz)
        t = np.array([tx, ty, tz])

        images.append(
            CameraPose(
                image_path=Path(image_name),
                rotation=R,
                translation=t,
            )
        )

    return ImageDataset(images=images, intrinsics=intrinsics, source_format="colmap")


def _parse_colmap_cameras_txt(path: Path) -> CameraIntrinsics:
    """Parse first camera from COLMAP cameras.txt."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS...
            model = parts[1]
            width, height = int(parts[2]), int(parts[3])
            params = [float(p) for p in parts[4:]]

            if model in ("SIMPLE_PINHOLE", "SIMPLE_RADIAL"):
                fx = fy = params[0]
                cx, cy = params[1], params[2]
            elif model in ("PINHOLE", "OPENCV"):
                fx, fy = params[0], params[1]
                cx, cy = params[2], params[3]
            else:
                fx = fy = params[0]
                cx, cy = width / 2, height / 2

            return CameraIntrinsics(
                fx=fx, fy=fy, cx=cx, cy=cy, width=width, height=height
            )

    raise ValueError(f"No camera found in {path}")


def _load_colmap_binary(sparse_dir: Path) -> ImageDataset:
    """Load COLMAP binary format."""
    intrinsics = _parse_colmap_cameras_bin(sparse_dir / "cameras.bin")

    images: list[CameraPose] = []
    with open(sparse_dir / "images.bin", "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            _image_id = struct.unpack("<I", f.read(4))[0]
            qw, qx, qy, qz = struct.unpack("<dddd", f.read(32))
            tx, ty, tz = struct.unpack("<ddd", f.read(24))
            _camera_id = struct.unpack("<I", f.read(4))[0]

            # Read name (null-terminated)
            name_bytes = b""
            while True:
                c = f.read(1)
                if c == b"\x00":
                    break
                name_bytes += c

            # Skip 2D points
            num_points2d = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2d * 24)  # x, y, point3d_id per point

            R = _quat_to_rot(qw, qx, qy, qz)
            images.append(
                CameraPose(
                    image_path=Path(name_bytes.decode("utf-8")),
                    rotation=R,
                    translation=np.array([tx, ty, tz]),
                )
            )

    return ImageDataset(images=images, intrinsics=intrinsics, source_format="colmap")


def _parse_colmap_cameras_bin(path: Path) -> CameraIntrinsics:
    """Parse first camera from COLMAP cameras.bin."""
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        if num_cameras == 0:
            raise ValueError("No cameras in cameras.bin")

        camera_id = struct.unpack("<I", f.read(4))[0]
        model_id = struct.unpack("<i", f.read(4))[0]
        width = struct.unpack("<Q", f.read(8))[0]
        height = struct.unpack("<Q", f.read(8))[0]

        # Number of params depends on model
        num_params = {0: 3, 1: 4, 2: 4, 3: 5, 4: 4, 5: 5}.get(model_id, 4)
        params = struct.unpack(f"<{num_params}d", f.read(num_params * 8))

        if model_id in (0, 3):  # SIMPLE_PINHOLE, SIMPLE_RADIAL
            fx = fy = params[0]
            cx, cy = params[1], params[2]
        else:  # PINHOLE, OPENCV, etc.
            fx, fy = params[0], params[1]
            cx, cy = params[2], params[3]

        return CameraIntrinsics(
            fx=fx, fy=fy, cx=cx, cy=cy, width=int(width), height=int(height)
        )


def load_nerfstudio_poses(transforms_json: Path) -> ImageDataset:
    """Parse nerfstudio transforms.json into ImageDataset."""
    with open(transforms_json) as f:
        data = json.load(f)

    # Extract intrinsics
    fx = data.get("fl_x", data.get("fx", 0))
    fy = data.get("fl_y", data.get("fy", fx))
    cx = data.get("cx", data.get("w", 0) / 2)
    cy = data.get("cy", data.get("h", 0) / 2)
    w = int(data.get("w", data.get("width", 0)))
    h = int(data.get("h", data.get("height", 0)))

    intrinsics = CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, width=w, height=h)

    images: list[CameraPose] = []
    for frame in data.get("frames", []):
        mat = np.array(frame["transform_matrix"], dtype=np.float64)
        R = mat[:3, :3]
        t = mat[:3, 3]
        images.append(
            CameraPose(
                image_path=Path(frame["file_path"]),
                rotation=R,
                translation=t,
                timestamp=frame.get("timestamp"),
            )
        )

    return ImageDataset(images=images, intrinsics=intrinsics, source_format="nerfstudio")


def load_lidar_point_cloud(
    lidar_dir: Path,
    extensions: list[str] | None = None,
) -> PointCloudData:
    """Load LiDAR point cloud data from directory (PLY, PCD, BIN, LAS)."""
    import open3d as o3d  # type: ignore[import-untyped]

    extensions = extensions or [".ply", ".pcd", ".bin"]

    files = sorted(
        f for f in lidar_dir.iterdir() if f.suffix.lower() in extensions
    )

    if not files:
        raise FileNotFoundError(
            f"No LiDAR files found in {lidar_dir} with extensions {extensions}"
        )

    all_points: list[np.ndarray] = []
    all_colors: list[np.ndarray] = []

    for f in files:
        if f.suffix.lower() == ".bin":
            # KITTI format: N x 4 float32 (x, y, z, intensity)
            data = np.fromfile(str(f), dtype=np.float32).reshape(-1, 4)
            all_points.append(data[:, :3].astype(np.float64))
        else:
            pcd = o3d.io.read_point_cloud(str(f))
            pts = np.asarray(pcd.points)
            all_points.append(pts)
            if pcd.has_colors():
                all_colors.append(np.asarray(pcd.colors))

    points = np.concatenate(all_points, axis=0)
    colors = np.concatenate(all_colors, axis=0) if all_colors else None

    return PointCloudData(points=points, colors=colors)


def _quat_to_rot(qw: float, qx: float, qy: float, qz: float) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    R = np.array(
        [
            [
                1 - 2 * (qy**2 + qz**2),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx**2 + qz**2),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx**2 + qy**2),
            ],
        ]
    )
    return R

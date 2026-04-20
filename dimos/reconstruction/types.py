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

"""Core data types for the 3D reconstruction pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    distortion_coeffs: NDArray[np.float64] | None = None  # k1, k2, p1, p2, ...


@dataclass(frozen=True)
class CameraPose:
    """Camera extrinsic: world-from-camera SE(3)."""

    image_path: Path
    rotation: NDArray[np.float64]  # (3,3) rotation matrix
    translation: NDArray[np.float64]  # (3,) translation vector
    timestamp: float | None = None


@dataclass
class ImageDataset:
    """Collection of images with known/estimated poses and intrinsics."""

    images: list[CameraPose]
    intrinsics: CameraIntrinsics
    source_format: str = "colmap"  # "colmap", "nerfstudio", "hloc"


@dataclass
class PointCloudData:
    """A point cloud with optional per-point attributes."""

    points: NDArray[np.float64]  # (N, 3)
    colors: NDArray[np.float64] | None = None  # (N, 3) in [0, 1]
    normals: NDArray[np.float64] | None = None  # (N, 3)
    intensities: NDArray[np.float64] | None = None  # (N,) — LiDAR

    @property
    def num_points(self) -> int:
        return self.points.shape[0]

    def subsample(self, n: int, rng: np.random.Generator | None = None) -> PointCloudData:
        """Randomly subsample to n points."""
        if self.num_points <= n:
            return self
        rng = rng or np.random.default_rng(42)
        idx = rng.choice(self.num_points, size=n, replace=False)
        return PointCloudData(
            points=self.points[idx],
            colors=self.colors[idx] if self.colors is not None else None,
            normals=self.normals[idx] if self.normals is not None else None,
            intensities=self.intensities[idx] if self.intensities is not None else None,
        )


@dataclass
class TriangleMesh:
    """Triangle mesh with optional per-vertex attributes."""

    vertices: NDArray[np.float64]  # (V, 3)
    faces: NDArray[np.int64]  # (F, 3)
    vertex_colors: NDArray[np.float64] | None = None  # (V, 3)
    vertex_normals: NDArray[np.float64] | None = None  # (V, 3)
    face_normals: NDArray[np.float64] | None = None  # (F, 3)

    @property
    def num_vertices(self) -> int:
        return self.vertices.shape[0]

    @property
    def num_faces(self) -> int:
        return self.faces.shape[0]


class AlignmentMethod(Enum):
    """Alignment methods for registering two reconstructions."""

    UMEYAMA_SIM3 = auto()
    ICP_POINT_TO_POINT = auto()
    ICP_POINT_TO_PLANE = auto()
    SHARED_TRAJECTORY = auto()
    MANUAL = auto()


@dataclass(frozen=True)
class Sim3Transform:
    """Similarity transform: scale * R @ x + t."""

    rotation: NDArray[np.float64]  # (3,3)
    translation: NDArray[np.float64]  # (3,)
    scale: float = 1.0

    def apply(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Apply Sim(3) to (N,3) points."""
        return self.scale * (points @ self.rotation.T) + self.translation

    @staticmethod
    def identity() -> Sim3Transform:
        return Sim3Transform(
            rotation=np.eye(3),
            translation=np.zeros(3),
            scale=1.0,
        )


@dataclass
class ReconstructionResult:
    """Output of a single reconstruction stage."""

    point_cloud: PointCloudData
    mesh: TriangleMesh | None = None
    is_metric: bool = False
    source: str = ""  # "colmap", "nerfacto", "3dgs", "lidar_poisson", etc.
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    """Output of the alignment stage."""

    transform: Sim3Transform
    method: AlignmentMethod
    aligned_reconstruction: ReconstructionResult
    residual_rmse: float = 0.0
    num_correspondences: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusedScene:
    """The merged scene after alignment + fusion."""

    point_cloud: PointCloudData
    mesh: TriangleMesh | None = None
    sources: list[ReconstructionResult] = field(default_factory=list)
    alignment: AlignmentResult | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class MujocoExport:
    """Exported simulation assets."""

    xml_path: Path
    xml_content: str
    mesh_paths: list[Path]
    collision_mesh_paths: list[Path]
    metadata: dict[str, Any] = field(default_factory=dict)

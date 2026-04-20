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

"""Pipeline configuration for 3D scene reconstruction."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field

from dimos.protocol.service.spec import BaseConfig
from dimos.reconstruction.types import AlignmentMethod


class IngestConfig(BaseConfig):
    """Configuration for data ingestion."""

    image_dir: Path | None = None
    lidar_dir: Path | None = None
    pose_file: Path | None = None
    pose_format: str = "colmap"  # "colmap", "nerfstudio", "custom"
    image_extensions: list[str] = Field(
        default_factory=lambda: [".jpg", ".png", ".jpeg"]
    )
    downsample_factor: int = 1


class ImageReconConfig(BaseConfig):
    """Configuration for image-based reconstruction."""

    method: str = "colmap"  # "colmap", "nerfacto", "3dgs", "openmvs"
    output_dir: Path = Path("output/image_recon")
    # COLMAP
    colmap_quality: str = "medium"  # "low", "medium", "high", "extreme"
    colmap_use_gpu: bool = True
    # NeRF / 3DGS
    num_iterations: int = 30000
    extract_mesh: bool = True
    mesh_resolution: int = 256
    extra_args: dict[str, Any] = Field(default_factory=dict)


class LidarReconConfig(BaseConfig):
    """Configuration for LiDAR point cloud → mesh reconstruction."""

    method: str = "poisson"  # "poisson", "ball_pivoting", "alpha_shape"
    voxel_downsample_size: float = 0.02
    estimate_normals: bool = True
    normal_radius: float = 0.1
    normal_max_nn: int = 30
    # Poisson
    poisson_depth: int = 9
    poisson_density_quantile: float = 0.01
    # Outlier removal
    remove_outliers: bool = True
    outlier_nb_neighbors: int = 20
    outlier_std_ratio: float = 2.0


class AlignmentConfig(BaseConfig):
    """Configuration for aligning image-based and LiDAR reconstructions."""

    method: AlignmentMethod = AlignmentMethod.ICP_POINT_TO_PLANE
    min_correspondences: int = 4
    # ICP
    icp_max_iterations: int = 200
    icp_threshold: float = 0.05
    # FPFH for coarse initial alignment
    use_fpfh_for_initial: bool = True
    fpfh_voxel_size: float = 0.05
    # Manual override
    manual_scale: float | None = None
    manual_transform: list[float] | None = None  # flat 4x4


class FusionConfig(BaseConfig):
    """Configuration for merging aligned reconstructions."""

    method: str = "merge"  # "merge", "tsdf", "weighted"
    voxel_downsample_after_merge: float = 0.01
    reconstruct_mesh: bool = True
    mesh_method: str = "poisson"
    mesh_poisson_depth: int = 10
    simplify_mesh: bool = True
    target_face_count: int = 500_000


class ExportConfig(BaseConfig):
    """Configuration for simulation export."""

    output_dir: Path = Path("output/export")
    export_mujoco_xml: bool = True
    export_obj: bool = True
    export_collision: bool = True
    collision_face_count: int = 10_000
    collision_convex_decomposition: bool = True
    max_convex_hulls: int = 32
    mujoco_model_name: str = "reconstructed_scene"
    floor_z_offset: float = 0.0


class EvalConfig(BaseConfig):
    """Configuration for evaluation metrics."""

    ground_truth_mesh_path: Path | None = None
    ground_truth_pointcloud_path: Path | None = None
    chamfer_num_samples: int = 100_000
    hausdorff_num_samples: int = 50_000
    completeness_voxel_size: float = 0.05
    mujoco_stability_steps: int = 1000
    mujoco_stability_max_penetration: float = 0.01


class PipelineConfig(BaseConfig):
    """Top-level reconstruction pipeline configuration."""

    run_name: str = "default"
    output_dir: Path = Path("output/reconstruction")
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    image_recon: ImageReconConfig = Field(default_factory=ImageReconConfig)
    lidar_recon: LidarReconConfig = Field(default_factory=LidarReconConfig)
    alignment: AlignmentConfig = Field(default_factory=AlignmentConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    eval: EvalConfig = Field(default_factory=EvalConfig)
    # Stage toggles
    skip_image_recon: bool = False
    skip_lidar_recon: bool = False
    skip_alignment: bool = False

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

"""Image-based reconstruction: COLMAP, NeRF, 3DGS backends."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]

from dimos.reconstruction.config import ImageReconConfig
from dimos.reconstruction.types import (
    ImageDataset,
    PointCloudData,
    ReconstructionResult,
    TriangleMesh,
)

logger = logging.getLogger(__name__)


class ImageReconStage:
    """Run image-based reconstruction. Dispatches to method-specific backends."""

    def __init__(self, config: ImageReconConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "image_recon"

    def run(self, input_data: ImageDataset) -> ReconstructionResult:
        if self.config.method == "colmap":
            return run_colmap_reconstruction(input_data, self.config)
        elif self.config.method == "nerfacto":
            return extract_point_cloud_from_nerf(input_data, self.config)
        elif self.config.method == "3dgs":
            return run_3dgs_reconstruction(input_data, self.config)
        else:
            raise ValueError(f"Unknown image recon method: {self.config.method}")

    def validate_input(self, input_data: ImageDataset) -> list[str]:
        errors: list[str] = []
        if len(input_data.images) < 3:
            errors.append(f"Need at least 3 images, got {len(input_data.images)}")
        return errors


# --- Pure functions ---


def run_colmap_reconstruction(
    dataset: ImageDataset,
    config: ImageReconConfig,
) -> ReconstructionResult:
    """Run COLMAP SfM + MVS pipeline, extract dense point cloud and mesh.

    Requires COLMAP to be installed and available in PATH.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    image_dir = dataset.images[0].image_path.parent if dataset.images else output_dir
    database_path = output_dir / "database.db"
    sparse_dir = output_dir / "sparse"
    dense_dir = output_dir / "dense"

    sparse_dir.mkdir(parents=True, exist_ok=True)
    dense_dir.mkdir(parents=True, exist_ok=True)

    quality_map = {
        "low": "--SiftExtraction.max_image_size 1000",
        "medium": "--SiftExtraction.max_image_size 2000",
        "high": "--SiftExtraction.max_image_size 3200",
        "extreme": "--SiftExtraction.max_image_size 4000",
    }
    quality_args = quality_map.get(config.colmap_quality, "")
    gpu_flag = "--SiftExtraction.use_gpu 1" if config.colmap_use_gpu else "--SiftExtraction.use_gpu 0"

    commands = [
        f"colmap feature_extractor --database_path {database_path} --image_path {image_dir} {quality_args} {gpu_flag}",
        f"colmap exhaustive_matcher --database_path {database_path}",
        f"colmap mapper --database_path {database_path} --image_path {image_dir} --output_path {sparse_dir}",
        f"colmap image_undistorter --image_path {image_dir} --input_path {sparse_dir}/0 --output_path {dense_dir}",
        f"colmap patch_match_stereo --workspace_path {dense_dir}",
        f"colmap stereo_fusion --workspace_path {dense_dir} --output_path {dense_dir}/fused.ply",
    ]

    for cmd in commands:
        logger.info("Running: %s", cmd)
        result = subprocess.run(
            cmd.split(), capture_output=True, text=True, check=False
        )
        if result.returncode != 0:
            logger.error("COLMAP command failed: %s\n%s", cmd, result.stderr)
            raise RuntimeError(f"COLMAP failed: {cmd}\n{result.stderr}")

    # Load fused point cloud
    fused_path = dense_dir / "fused.ply"
    pcd = o3d.io.read_point_cloud(str(fused_path))

    point_cloud = PointCloudData(
        points=np.asarray(pcd.points),
        colors=np.asarray(pcd.colors) if pcd.has_colors() else None,
    )

    # Optional mesh extraction via Poisson
    mesh = None
    if config.extract_mesh:
        pcd.estimate_normals()
        o3d_mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=8
        )
        o3d_mesh.compute_vertex_normals()
        mesh = TriangleMesh(
            vertices=np.asarray(o3d_mesh.vertices),
            faces=np.asarray(o3d_mesh.triangles).astype(np.int64),
            vertex_colors=np.asarray(o3d_mesh.vertex_colors) if o3d_mesh.has_vertex_colors() else None,
        )

    return ReconstructionResult(
        point_cloud=point_cloud,
        mesh=mesh,
        is_metric=False,  # COLMAP is up-to-scale without GPS/known baseline
        source="colmap",
        metadata={"quality": config.colmap_quality, "output_dir": str(output_dir)},
    )


def extract_point_cloud_from_nerf(
    dataset: ImageDataset,
    config: ImageReconConfig,
) -> ReconstructionResult:
    """Train a NeRF model using nerfstudio, extract point cloud.

    Requires nerfstudio to be installed: pip install nerfstudio.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train
    train_cmd = [
        "ns-train", "nerfacto",
        "--data", str(dataset.images[0].image_path.parent),
        "--output-dir", str(output_dir),
        "--max-num-iterations", str(config.num_iterations),
        "--viewer.quit-on-train-completion", "True",
    ]
    for k, v in config.extra_args.items():
        train_cmd.extend([f"--{k}", str(v)])

    logger.info("Running: %s", " ".join(train_cmd))
    result = subprocess.run(train_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Nerfstudio training failed:\n{result.stderr}")

    # Export point cloud
    export_cmd = [
        "ns-export", "pointcloud",
        "--load-config", str(output_dir / "nerfacto" / "config.yml"),
        "--output-dir", str(output_dir / "exports"),
        "--num-points", str(config.chamfer_num_samples if hasattr(config, "chamfer_num_samples") else 1000000),
    ]

    result = subprocess.run(export_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Nerfstudio export failed:\n{result.stderr}")

    # Load exported point cloud
    export_path = output_dir / "exports" / "point_cloud.ply"
    pcd = o3d.io.read_point_cloud(str(export_path))

    return ReconstructionResult(
        point_cloud=PointCloudData(
            points=np.asarray(pcd.points),
            colors=np.asarray(pcd.colors) if pcd.has_colors() else None,
        ),
        is_metric=False,
        source="nerfacto",
        metadata={"num_iterations": config.num_iterations},
    )


def run_3dgs_reconstruction(
    dataset: ImageDataset,
    config: ImageReconConfig,
) -> ReconstructionResult:
    """Train 3D Gaussian Splatting, extract point cloud from Gaussian centers.

    Requires nerfstudio with gsplat: pip install nerfstudio gsplat.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        "ns-train", "splatfacto",
        "--data", str(dataset.images[0].image_path.parent),
        "--output-dir", str(output_dir),
        "--max-num-iterations", str(config.num_iterations),
        "--viewer.quit-on-train-completion", "True",
    ]

    logger.info("Running: %s", " ".join(train_cmd))
    result = subprocess.run(train_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"3DGS training failed:\n{result.stderr}")

    # Export
    export_cmd = [
        "ns-export", "pointcloud",
        "--load-config", str(output_dir / "splatfacto" / "config.yml"),
        "--output-dir", str(output_dir / "exports"),
    ]

    result = subprocess.run(export_cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"3DGS export failed:\n{result.stderr}")

    export_path = output_dir / "exports" / "point_cloud.ply"
    pcd = o3d.io.read_point_cloud(str(export_path))

    return ReconstructionResult(
        point_cloud=PointCloudData(
            points=np.asarray(pcd.points),
            colors=np.asarray(pcd.colors) if pcd.has_colors() else None,
        ),
        is_metric=False,
        source="3dgs",
        metadata={"num_iterations": config.num_iterations},
    )

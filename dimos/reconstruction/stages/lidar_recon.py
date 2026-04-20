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

"""LiDAR point cloud → mesh reconstruction."""

from __future__ import annotations

import logging

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]

from dimos.reconstruction.config import LidarReconConfig
from dimos.reconstruction.types import PointCloudData, ReconstructionResult, TriangleMesh

logger = logging.getLogger(__name__)


class LidarReconStage:
    """Reconstruct mesh from LiDAR point cloud."""

    def __init__(self, config: LidarReconConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "lidar_recon"

    def run(self, input_data: PointCloudData) -> ReconstructionResult:
        preprocessed = preprocess_point_cloud(
            input_data,
            voxel_size=self.config.voxel_downsample_size,
            remove_outliers=self.config.remove_outliers,
            outlier_nb_neighbors=self.config.outlier_nb_neighbors,
            outlier_std_ratio=self.config.outlier_std_ratio,
        )

        if self.config.estimate_normals and preprocessed.normals is None:
            preprocessed = _estimate_normals(
                preprocessed,
                radius=self.config.normal_radius,
                max_nn=self.config.normal_max_nn,
            )

        if self.config.method == "poisson":
            return poisson_reconstruction(
                preprocessed,
                depth=self.config.poisson_depth,
                density_quantile=self.config.poisson_density_quantile,
            )
        elif self.config.method == "ball_pivoting":
            return ball_pivoting_reconstruction(preprocessed)
        else:
            raise ValueError(f"Unknown LiDAR reconstruction method: {self.config.method}")

    def validate_input(self, input_data: PointCloudData) -> list[str]:
        errors: list[str] = []
        if input_data.num_points < 100:
            errors.append(f"Too few points: {input_data.num_points}")
        return errors


# --- Pure functions ---


def preprocess_point_cloud(
    pcd: PointCloudData,
    voxel_size: float,
    remove_outliers: bool = True,
    outlier_nb_neighbors: int = 20,
    outlier_std_ratio: float = 2.0,
) -> PointCloudData:
    """Downsample, remove outliers."""
    o3d_pcd = _to_o3d(pcd)

    # Voxel downsample
    o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_size)

    # Statistical outlier removal
    if remove_outliers:
        o3d_pcd, _ = o3d_pcd.remove_statistical_outlier(
            nb_neighbors=outlier_nb_neighbors,
            std_ratio=outlier_std_ratio,
        )

    return _from_o3d(o3d_pcd)


def poisson_reconstruction(
    pcd: PointCloudData,
    depth: int = 9,
    density_quantile: float = 0.01,
) -> ReconstructionResult:
    """Screened Poisson surface reconstruction via Open3D."""
    o3d_pcd = _to_o3d(pcd)

    if not o3d_pcd.has_normals():
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        o3d_pcd.orient_normals_consistent_tangent_plane(k=15)

    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        o3d_pcd, depth=depth
    )

    # Remove low-density vertices
    densities_arr = np.asarray(densities)
    threshold = np.quantile(densities_arr, density_quantile)
    vertices_to_remove = densities_arr < threshold
    mesh.remove_vertices_by_mask(vertices_to_remove)

    tri_mesh = _o3d_mesh_to_triangle_mesh(mesh)
    result_pcd = _from_o3d(o3d_pcd)

    return ReconstructionResult(
        point_cloud=result_pcd,
        mesh=tri_mesh,
        is_metric=True,
        source="lidar_poisson",
        metadata={"poisson_depth": depth, "density_quantile": density_quantile},
    )


def ball_pivoting_reconstruction(
    pcd: PointCloudData,
    radii: list[float] | None = None,
) -> ReconstructionResult:
    """Ball-pivoting algorithm for mesh reconstruction."""
    o3d_pcd = _to_o3d(pcd)

    if not o3d_pcd.has_normals():
        o3d_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )

    if radii is None:
        # Auto-estimate radii from average point spacing
        dists = o3d_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dists)
        radii = [avg_dist * f for f in [0.5, 1.0, 2.0, 4.0]]

    radii_vec = o3d.utility.DoubleVector(radii)
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        o3d_pcd, radii_vec
    )

    tri_mesh = _o3d_mesh_to_triangle_mesh(mesh)
    result_pcd = _from_o3d(o3d_pcd)

    return ReconstructionResult(
        point_cloud=result_pcd,
        mesh=tri_mesh,
        is_metric=True,
        source="lidar_ball_pivoting",
        metadata={"radii": radii},
    )


# --- Helpers ---


def _estimate_normals(
    pcd: PointCloudData, radius: float = 0.1, max_nn: int = 30
) -> PointCloudData:
    """Estimate point normals using Open3D."""
    o3d_pcd = _to_o3d(pcd)
    o3d_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn)
    )
    o3d_pcd.orient_normals_consistent_tangent_plane(k=15)
    return _from_o3d(o3d_pcd)


def _to_o3d(pcd: PointCloudData) -> o3d.geometry.PointCloud:
    """Convert PointCloudData to Open3D PointCloud."""
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
    if pcd.normals is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals)
    return o3d_pcd


def _from_o3d(o3d_pcd: o3d.geometry.PointCloud) -> PointCloudData:
    """Convert Open3D PointCloud to PointCloudData."""
    points = np.asarray(o3d_pcd.points)
    colors = np.asarray(o3d_pcd.colors) if o3d_pcd.has_colors() else None
    normals = np.asarray(o3d_pcd.normals) if o3d_pcd.has_normals() else None
    return PointCloudData(points=points, colors=colors, normals=normals)


def _o3d_mesh_to_triangle_mesh(mesh: o3d.geometry.TriangleMesh) -> TriangleMesh:
    """Convert Open3D TriangleMesh to our TriangleMesh type."""
    mesh.compute_vertex_normals()
    return TriangleMesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles).astype(np.int64),
        vertex_normals=np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None,
        vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
    )

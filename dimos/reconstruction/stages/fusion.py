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

"""Merge aligned reconstructions into a single fused scene."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]

from dimos.reconstruction.config import FusionConfig
from dimos.reconstruction.types import (
    FusedScene,
    PointCloudData,
    ReconstructionResult,
    TriangleMesh,
)

logger = logging.getLogger(__name__)


@dataclass
class FusionInput:
    """Input for the fusion stage."""

    reconstructions: list[ReconstructionResult]


class FusionStage:
    """Merge aligned reconstructions into a single scene."""

    def __init__(self, config: FusionConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "fusion"

    def run(self, input_data: FusionInput) -> FusedScene:
        clouds = [r.point_cloud for r in input_data.reconstructions]

        merged_pcd = merge_point_clouds(
            clouds,
            voxel_downsample_size=self.config.voxel_downsample_after_merge,
        )

        mesh = None
        if self.config.reconstruct_mesh:
            mesh = reconstruct_mesh_from_fused(
                merged_pcd,
                method=self.config.mesh_method,
                poisson_depth=self.config.mesh_poisson_depth,
            )
            if self.config.simplify_mesh and mesh is not None:
                mesh = simplify_mesh(mesh, self.config.target_face_count)

        return FusedScene(
            point_cloud=merged_pcd,
            mesh=mesh,
            sources=input_data.reconstructions,
        )

    def validate_input(self, input_data: FusionInput) -> list[str]:
        errors: list[str] = []
        if len(input_data.reconstructions) == 0:
            errors.append("No reconstructions to fuse")
        return errors


# --- Pure functions ---


def merge_point_clouds(
    clouds: list[PointCloudData],
    voxel_downsample_size: float = 0.01,
) -> PointCloudData:
    """Concatenate and voxel-downsample multiple point clouds."""
    all_points = np.concatenate([c.points for c in clouds], axis=0)

    all_colors = None
    color_clouds = [c for c in clouds if c.colors is not None]
    if color_clouds:
        # For clouds without colors, fill with gray
        color_parts = []
        for c in clouds:
            if c.colors is not None:
                color_parts.append(c.colors)
            else:
                color_parts.append(np.full((c.num_points, 3), 0.5))
        all_colors = np.concatenate(color_parts, axis=0)

    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(all_points)
    if all_colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(all_colors)

    o3d_pcd = o3d_pcd.voxel_down_sample(voxel_size=voxel_downsample_size)

    points = np.asarray(o3d_pcd.points)
    colors = np.asarray(o3d_pcd.colors) if o3d_pcd.has_colors() else None

    return PointCloudData(points=points, colors=colors)


def reconstruct_mesh_from_fused(
    pcd: PointCloudData,
    method: str = "poisson",
    poisson_depth: int = 10,
) -> TriangleMesh | None:
    """Surface reconstruction on the fused point cloud."""
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)

    o3d_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    o3d_pcd.orient_normals_consistent_tangent_plane(k=15)

    if method == "poisson":
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            o3d_pcd, depth=poisson_depth
        )
        # Remove low-density vertices
        densities_arr = np.asarray(densities)
        threshold = np.quantile(densities_arr, 0.01)
        mesh.remove_vertices_by_mask(densities_arr < threshold)
    elif method == "ball_pivoting":
        dists = o3d_pcd.compute_nearest_neighbor_distance()
        avg_dist = np.mean(dists)
        radii = o3d.utility.DoubleVector(
            [avg_dist * f for f in [0.5, 1.0, 2.0, 4.0]]
        )
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            o3d_pcd, radii
        )
    else:
        logger.warning("Unknown mesh method %s, skipping", method)
        return None

    mesh.compute_vertex_normals()

    return TriangleMesh(
        vertices=np.asarray(mesh.vertices),
        faces=np.asarray(mesh.triangles).astype(np.int64),
        vertex_colors=np.asarray(mesh.vertex_colors) if mesh.has_vertex_colors() else None,
        vertex_normals=np.asarray(mesh.vertex_normals) if mesh.has_vertex_normals() else None,
    )


def simplify_mesh(
    mesh: TriangleMesh,
    target_face_count: int,
) -> TriangleMesh:
    """Quadric edge collapse mesh simplification."""
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    if mesh.vertex_colors is not None:
        o3d_mesh.vertex_colors = o3d.utility.Vector3dVector(mesh.vertex_colors)

    simplified = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=target_face_count
    )
    simplified.compute_vertex_normals()

    return TriangleMesh(
        vertices=np.asarray(simplified.vertices),
        faces=np.asarray(simplified.triangles).astype(np.int64),
        vertex_colors=np.asarray(simplified.vertex_colors) if simplified.has_vertex_colors() else None,
        vertex_normals=np.asarray(simplified.vertex_normals) if simplified.has_vertex_normals() else None,
    )

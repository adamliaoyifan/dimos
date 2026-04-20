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

"""Scale alignment between image-based (up-to-scale) and LiDAR (metric) reconstructions."""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import open3d as o3d  # type: ignore[import-untyped]
from numpy.typing import NDArray

from dimos.reconstruction.config import AlignmentConfig
from dimos.reconstruction.types import (
    AlignmentMethod,
    AlignmentResult,
    PointCloudData,
    ReconstructionResult,
    Sim3Transform,
    TriangleMesh,
)

logger = logging.getLogger(__name__)


@dataclass
class AlignmentInput:
    """Input for the alignment stage."""

    source: ReconstructionResult  # To be transformed (e.g., image-based)
    target: ReconstructionResult  # Reference frame (e.g., LiDAR, metric)


class AlignmentStage:
    """Align two reconstructions with potentially different scales."""

    def __init__(self, config: AlignmentConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "alignment"

    def run(self, input_data: AlignmentInput) -> AlignmentResult:
        source_pcd = input_data.source.point_cloud
        target_pcd = input_data.target.point_cloud

        if self.config.method == AlignmentMethod.SHARED_TRAJECTORY:
            # No alignment needed — already in same frame
            transform = Sim3Transform.identity()
        elif self.config.method == AlignmentMethod.MANUAL:
            if self.config.manual_scale is None:
                raise ValueError("Manual alignment requires manual_scale")
            transform = Sim3Transform(
                rotation=np.eye(3),
                translation=np.zeros(3),
                scale=self.config.manual_scale,
            )
        elif self.config.method == AlignmentMethod.UMEYAMA_SIM3:
            raise ValueError(
                "Umeyama requires explicit correspondences. "
                "Use umeyama_sim3() directly with correspondence pairs."
            )
        else:
            # ICP-based alignment with optional FPFH coarse initialization
            transform = _icp_pipeline(
                source_pcd,
                target_pcd,
                config=self.config,
            )

        aligned_recon = apply_transform_to_reconstruction(
            input_data.source, transform
        )

        # Compute residual
        aligned_pts = aligned_recon.point_cloud.subsample(10000).points
        target_pts = target_pcd.subsample(10000).points
        from scipy.spatial import KDTree  # type: ignore[import-untyped]

        tree = KDTree(target_pts)
        dists, _ = tree.query(aligned_pts, k=1)
        residual_rmse = float(np.sqrt(np.mean(dists**2)))

        return AlignmentResult(
            transform=transform,
            method=self.config.method,
            aligned_reconstruction=aligned_recon,
            residual_rmse=residual_rmse,
            num_correspondences=len(aligned_pts),
        )

    def validate_input(self, input_data: AlignmentInput) -> list[str]:
        errors: list[str] = []
        if input_data.source.point_cloud.num_points < 10:
            errors.append("Source has too few points")
        if input_data.target.point_cloud.num_points < 10:
            errors.append("Target has too few points")
        return errors


# --- Pure functions ---


def umeyama_sim3(
    source_points: NDArray[np.float64],
    target_points: NDArray[np.float64],
) -> Sim3Transform:
    """Compute Sim(3) transform via Umeyama's method.

    Given N >= 3 corresponding point pairs, finds (s, R, t) minimizing
    || s*R*source + t - target ||^2.
    """
    assert source_points.shape == target_points.shape
    n, dim = source_points.shape
    assert n >= 3 and dim == 3

    mu_src = source_points.mean(axis=0)
    mu_dst = target_points.mean(axis=0)

    src_centered = source_points - mu_src
    dst_centered = target_points - mu_dst

    H = src_centered.T @ dst_centered / n
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, np.sign(d)])

    R = Vt.T @ D @ U.T

    var_src = np.sum(src_centered**2) / n
    scale = float(np.trace(np.diag(S) @ D) / var_src)

    t = mu_dst - scale * R @ mu_src

    return Sim3Transform(rotation=R, translation=t, scale=scale)


def icp_alignment(
    source_pcd: PointCloudData,
    target_pcd: PointCloudData,
    method: AlignmentMethod = AlignmentMethod.ICP_POINT_TO_PLANE,
    max_iterations: int = 200,
    threshold: float = 0.05,
    initial_transform: NDArray[np.float64] | None = None,
) -> AlignmentResult:
    """Iterative Closest Point alignment (SE(3) — no scale).

    For scale alignment, run umeyama_sim3 first or use the full AlignmentStage.
    """
    src_o3d = _to_o3d_pcd(source_pcd)
    tgt_o3d = _to_o3d_pcd(target_pcd)

    if method == AlignmentMethod.ICP_POINT_TO_PLANE:
        src_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        tgt_o3d.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
        )
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    init = initial_transform if initial_transform is not None else np.eye(4)

    result = o3d.pipelines.registration.registration_icp(
        src_o3d,
        tgt_o3d,
        threshold,
        init,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    T = result.transformation
    R = T[:3, :3]
    t = T[:3, 3]

    transform = Sim3Transform(rotation=R, translation=t, scale=1.0)
    aligned_pcd = PointCloudData(points=transform.apply(source_pcd.points))

    return AlignmentResult(
        transform=transform,
        method=method,
        aligned_reconstruction=ReconstructionResult(
            point_cloud=aligned_pcd, is_metric=True, source="icp_aligned"
        ),
        residual_rmse=float(result.inlier_rmse),
        num_correspondences=len(result.correspondence_set),
    )


def fpfh_global_registration(
    source_pcd: PointCloudData,
    target_pcd: PointCloudData,
    voxel_size: float = 0.05,
) -> NDArray[np.float64]:
    """FPFH feature-based global registration for initial alignment guess.

    Returns 4x4 transformation matrix.
    """
    src = _to_o3d_pcd(source_pcd)
    tgt = _to_o3d_pcd(target_pcd)

    src = src.voxel_down_sample(voxel_size)
    tgt = tgt.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    src.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )
    tgt.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30)
    )

    radius_feature = voxel_size * 5
    src_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        src, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )
    tgt_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        tgt, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        src,
        tgt,
        src_fpfh,
        tgt_fpfh,
        mutual_filter=True,
        max_correspondence_distance=voxel_size * 1.5,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        ransac_n=3,
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
    )

    return np.asarray(result.transformation)


def apply_transform_to_reconstruction(
    recon: ReconstructionResult,
    transform: Sim3Transform,
) -> ReconstructionResult:
    """Apply Sim(3) transform to all geometry in a reconstruction."""
    new_points = transform.apply(recon.point_cloud.points)
    new_pcd = PointCloudData(
        points=new_points,
        colors=recon.point_cloud.colors,
        normals=recon.point_cloud.normals,
        intensities=recon.point_cloud.intensities,
    )

    new_mesh = None
    if recon.mesh is not None:
        new_vertices = transform.apply(recon.mesh.vertices)
        new_mesh = TriangleMesh(
            vertices=new_vertices,
            faces=recon.mesh.faces.copy(),
            vertex_colors=recon.mesh.vertex_colors,
            vertex_normals=recon.mesh.vertex_normals,
            face_normals=recon.mesh.face_normals,
        )

    return ReconstructionResult(
        point_cloud=new_pcd,
        mesh=new_mesh,
        is_metric=True,  # After alignment to metric frame
        source=f"{recon.source}_aligned",
        metadata={**recon.metadata, "alignment_scale": transform.scale},
    )


# --- Internal helpers ---


def _icp_pipeline(
    source_pcd: PointCloudData,
    target_pcd: PointCloudData,
    config: AlignmentConfig,
) -> Sim3Transform:
    """Full ICP pipeline: coarse scale → FPFH → ICP refinement."""
    src_o3d = _to_o3d_pcd(source_pcd)
    tgt_o3d = _to_o3d_pcd(target_pcd)

    # Coarse scale via bounding box ratio
    src_bbox = src_o3d.get_axis_aligned_bounding_box()
    tgt_bbox = tgt_o3d.get_axis_aligned_bounding_box()
    src_extent = np.linalg.norm(src_bbox.get_extent())
    tgt_extent = np.linalg.norm(tgt_bbox.get_extent())

    coarse_scale = float(tgt_extent / src_extent) if src_extent > 1e-10 else 1.0

    # Apply scale
    scaled_points = source_pcd.points * coarse_scale
    scaled_pcd = PointCloudData(points=scaled_points)

    # FPFH global registration for initial rotation + translation
    init_transform = np.eye(4)
    if config.use_fpfh_for_initial:
        init_transform = fpfh_global_registration(
            scaled_pcd, target_pcd, voxel_size=config.fpfh_voxel_size
        )

    # ICP refinement
    src_scaled_o3d = _to_o3d_pcd(scaled_pcd)
    tgt_o3d_fresh = _to_o3d_pcd(target_pcd)

    src_scaled_o3d.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )
    tgt_o3d_fresh.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30)
    )

    if config.method == AlignmentMethod.ICP_POINT_TO_PLANE:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane()
    else:
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()

    result = o3d.pipelines.registration.registration_icp(
        src_scaled_o3d,
        tgt_o3d_fresh,
        config.icp_threshold,
        init_transform,
        estimation,
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=config.icp_max_iterations
        ),
    )

    T = result.transformation
    R = T[:3, :3]
    t = T[:3, 3]

    # Combine: first scale, then rigid transform
    # final = R @ (scale * x) + t = (scale * R) @ x + t
    combined_R = R  # Rotation after scaling
    combined_scale = coarse_scale  # Scale is applied before rotation

    return Sim3Transform(rotation=combined_R, translation=t, scale=combined_scale)


def _to_o3d_pcd(pcd: PointCloudData) -> o3d.geometry.PointCloud:
    """Convert PointCloudData to Open3D."""
    o3d_pcd = o3d.geometry.PointCloud()
    o3d_pcd.points = o3d.utility.Vector3dVector(pcd.points)
    if pcd.colors is not None:
        o3d_pcd.colors = o3d.utility.Vector3dVector(pcd.colors)
    if pcd.normals is not None:
        o3d_pcd.normals = o3d.utility.Vector3dVector(pcd.normals)
    return o3d_pcd

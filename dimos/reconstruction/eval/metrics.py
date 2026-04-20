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

"""Pure metric functions for reconstruction evaluation.

Each function takes typed inputs and returns a frozen dataclass result.
All are standalone, composable, and testable in isolation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.spatial import KDTree  # type: ignore[import-untyped]

from dimos.reconstruction.types import (
    MujocoExport,
    PointCloudData,
    Sim3Transform,
    TriangleMesh,
)

logger = logging.getLogger(__name__)


# ============================================================
# Result dataclasses
# ============================================================


@dataclass(frozen=True)
class GeometricAccuracyResult:
    """Geometric accuracy between reconstructed and ground-truth point clouds."""

    chamfer_distance: float
    chamfer_distance_std: float
    hausdorff_distance: float
    hausdorff_95: float
    rmse: float
    num_samples_source: int
    num_samples_target: int


@dataclass(frozen=True)
class AlignmentQualityResult:
    """Quality of alignment between two reconstructions."""

    residual_rmse: float
    residual_max: float
    scale_factor: float
    scale_consistency: float
    num_correspondences: int
    inlier_ratio: float


@dataclass(frozen=True)
class MeshQualityResult:
    """Intrinsic mesh quality (no ground truth needed)."""

    num_vertices: int
    num_faces: int
    is_watertight: bool
    num_non_manifold_edges: int
    num_degenerate_faces: int
    normal_consistency: float
    bounding_box_extent: NDArray[np.float64]
    surface_area: float


@dataclass(frozen=True)
class CompletenessResult:
    """Scene completeness via voxelized overlap."""

    coverage_ratio: float
    precision: float
    recall: float
    f_score: float
    voxel_size_used: float
    num_gt_voxels: int
    num_recon_voxels: int
    num_matched_voxels: int


@dataclass(frozen=True)
class SimulationReadinessResult:
    """Whether the exported scene is ready for MuJoCo simulation."""

    collision_geometry_valid: bool
    num_collision_bodies: int
    total_collision_faces: int
    physics_stable: bool
    max_penetration_depth: float
    simulation_steps_run: int
    xml_parses_ok: bool


# ============================================================
# Geometric Accuracy
# ============================================================


def _nn_distances(
    source: NDArray[np.float64], target: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Nearest-neighbor distances from every source point to target."""
    tree = KDTree(target)
    dists, _ = tree.query(source, k=1)
    return dists


def chamfer_distance(
    source: PointCloudData,
    target: PointCloudData,
    num_samples: int = 100_000,
) -> tuple[float, float]:
    """Symmetric Chamfer distance between two point clouds.

    Returns (mean, std) of bidirectional nearest-neighbor distances.
    """
    src = source.subsample(num_samples).points
    tgt = target.subsample(num_samples).points

    d_s2t = _nn_distances(src, tgt)
    d_t2s = _nn_distances(tgt, src)

    all_dists = np.concatenate([d_s2t, d_t2s])
    return float(np.mean(all_dists)), float(np.std(all_dists))


def hausdorff_distance(
    source: PointCloudData,
    target: PointCloudData,
    num_samples: int = 50_000,
    percentile: float = 100.0,
) -> float:
    """One-directional Hausdorff: max (or percentile) of min-dist source→target."""
    src = source.subsample(num_samples).points
    tgt = target.subsample(num_samples).points

    dists = _nn_distances(src, tgt)
    if percentile >= 100.0:
        return float(np.max(dists))
    return float(np.percentile(dists, percentile))


def point_cloud_rmse(
    source: PointCloudData,
    target: PointCloudData,
    num_samples: int = 100_000,
) -> float:
    """RMSE of nearest-neighbor distances from source to target."""
    src = source.subsample(num_samples).points
    tgt = target.subsample(num_samples).points

    dists = _nn_distances(src, tgt)
    return float(np.sqrt(np.mean(dists**2)))


def geometric_accuracy(
    source: PointCloudData,
    target: PointCloudData,
    num_chamfer_samples: int = 100_000,
    num_hausdorff_samples: int = 50_000,
) -> GeometricAccuracyResult:
    """Compute all geometric accuracy metrics in one call."""
    src_sub = source.subsample(num_chamfer_samples)
    tgt_sub = target.subsample(num_chamfer_samples)

    src_pts = src_sub.points
    tgt_pts = tgt_sub.points

    d_s2t = _nn_distances(src_pts, tgt_pts)
    d_t2s = _nn_distances(tgt_pts, src_pts)

    all_dists = np.concatenate([d_s2t, d_t2s])

    # Hausdorff on a (possibly smaller) subsample
    src_h = source.subsample(num_hausdorff_samples).points
    tgt_h = target.subsample(num_hausdorff_samples).points
    h_dists = _nn_distances(src_h, tgt_h)

    return GeometricAccuracyResult(
        chamfer_distance=float(np.mean(all_dists)),
        chamfer_distance_std=float(np.std(all_dists)),
        hausdorff_distance=float(np.max(h_dists)),
        hausdorff_95=float(np.percentile(h_dists, 95)),
        rmse=float(np.sqrt(np.mean(d_s2t**2))),
        num_samples_source=len(src_pts),
        num_samples_target=len(tgt_pts),
    )


# ============================================================
# Alignment Quality
# ============================================================


def alignment_residual(
    source_points: NDArray[np.float64],
    target_points: NDArray[np.float64],
    transform: Sim3Transform,
    inlier_threshold: float = 0.05,
) -> AlignmentQualityResult:
    """Evaluate alignment quality given known correspondences.

    Applies transform to source_points and measures residual vs target_points.
    """
    aligned = transform.apply(source_points)
    residuals = np.linalg.norm(aligned - target_points, axis=1)

    # Per-correspondence scale estimates
    src_dists = np.linalg.norm(source_points - source_points.mean(axis=0), axis=1)
    tgt_dists = np.linalg.norm(target_points - target_points.mean(axis=0), axis=1)
    # Avoid division by zero
    mask = src_dists > 1e-10
    if mask.any():
        per_scale = tgt_dists[mask] / src_dists[mask]
        scale_consistency = float(np.std(per_scale))
    else:
        scale_consistency = 0.0

    inliers = residuals < inlier_threshold
    return AlignmentQualityResult(
        residual_rmse=float(np.sqrt(np.mean(residuals**2))),
        residual_max=float(np.max(residuals)),
        scale_factor=transform.scale,
        scale_consistency=scale_consistency,
        num_correspondences=len(source_points),
        inlier_ratio=float(np.mean(inliers)),
    )


def alignment_quality_from_clouds(
    aligned_source: PointCloudData,
    target: PointCloudData,
    inlier_threshold: float = 0.05,
    num_samples: int = 50_000,
) -> AlignmentQualityResult:
    """Evaluate alignment quality without known correspondences.

    Uses nearest-neighbor matching to establish correspondences.
    """
    src = aligned_source.subsample(num_samples).points
    tgt = target.subsample(num_samples).points

    tree = KDTree(tgt)
    dists, indices = tree.query(src, k=1)
    dists = np.asarray(dists)

    inliers = dists < inlier_threshold
    return AlignmentQualityResult(
        residual_rmse=float(np.sqrt(np.mean(dists**2))),
        residual_max=float(np.max(dists)),
        scale_factor=1.0,  # Unknown without explicit correspondences
        scale_consistency=0.0,
        num_correspondences=len(src),
        inlier_ratio=float(np.mean(inliers)),
    )


# ============================================================
# Mesh Quality
# ============================================================


def mesh_quality(mesh: TriangleMesh) -> MeshQualityResult:
    """Evaluate intrinsic mesh quality (no ground truth needed)."""
    import open3d as o3d  # type: ignore[import-untyped]

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))
    o3d_mesh.compute_vertex_normals()

    # Watertightness
    is_watertight = o3d_mesh.is_watertight()

    # Non-manifold edges
    non_manifold_edges = np.asarray(o3d_mesh.get_non_manifold_edges(allow_boundary_edges=False))
    num_non_manifold = len(non_manifold_edges)

    # Degenerate faces (zero area)
    verts = mesh.vertices
    faces = mesh.faces
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    num_degenerate = int(np.sum(areas < 1e-10))
    total_area = float(np.sum(areas))

    # Normal consistency: fraction of adjacent faces with consistent orientation
    # Approximated by checking if computed normals are consistent
    if mesh.face_normals is not None:
        computed_normals = cross / (np.linalg.norm(cross, axis=1, keepdims=True) + 1e-10)
        dots = np.sum(computed_normals * mesh.face_normals, axis=1)
        consistency = float(np.mean(dots > 0))
    else:
        consistency = 1.0 if is_watertight else 0.5

    bbox = o3d_mesh.get_axis_aligned_bounding_box()
    extent = np.asarray(bbox.get_extent())

    return MeshQualityResult(
        num_vertices=mesh.num_vertices,
        num_faces=mesh.num_faces,
        is_watertight=is_watertight,
        num_non_manifold_edges=num_non_manifold,
        num_degenerate_faces=num_degenerate,
        normal_consistency=consistency,
        bounding_box_extent=extent,
        surface_area=total_area,
    )


# ============================================================
# Completeness
# ============================================================


def completeness(
    reconstructed: PointCloudData,
    ground_truth: PointCloudData,
    voxel_size: float = 0.05,
    distance_threshold: float | None = None,
) -> CompletenessResult:
    """Evaluate reconstruction completeness via voxelized overlap.

    Voxelizes both point clouds, then computes precision, recall, F-score.
    """
    if distance_threshold is not None:
        return _completeness_distance_based(
            reconstructed, ground_truth, distance_threshold
        )

    def _voxelize(pts: NDArray[np.float64]) -> set[tuple[int, int, int]]:
        indices = np.floor(pts / voxel_size).astype(np.int64)
        return {(int(r[0]), int(r[1]), int(r[2])) for r in indices}

    recon_voxels = _voxelize(reconstructed.points)
    gt_voxels = _voxelize(ground_truth.points)

    matched = recon_voxels & gt_voxels

    n_recon = len(recon_voxels)
    n_gt = len(gt_voxels)
    n_matched = len(matched)

    precision = n_matched / n_recon if n_recon > 0 else 0.0
    recall = n_matched / n_gt if n_gt > 0 else 0.0
    f_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return CompletenessResult(
        coverage_ratio=recall,
        precision=precision,
        recall=recall,
        f_score=f_score,
        voxel_size_used=voxel_size,
        num_gt_voxels=n_gt,
        num_recon_voxels=n_recon,
        num_matched_voxels=n_matched,
    )


def _completeness_distance_based(
    reconstructed: PointCloudData,
    ground_truth: PointCloudData,
    distance_threshold: float,
) -> CompletenessResult:
    """Distance-based completeness: count GT points within threshold of recon."""
    tree_recon = KDTree(reconstructed.points)
    dists_gt, _ = tree_recon.query(ground_truth.points, k=1)
    covered = dists_gt < distance_threshold

    tree_gt = KDTree(ground_truth.points)
    dists_recon, _ = tree_gt.query(reconstructed.points, k=1)
    correct = dists_recon < distance_threshold

    recall = float(np.mean(covered))
    precision = float(np.mean(correct))
    f_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return CompletenessResult(
        coverage_ratio=recall,
        precision=precision,
        recall=recall,
        f_score=f_score,
        voxel_size_used=distance_threshold,
        num_gt_voxels=ground_truth.num_points,
        num_recon_voxels=reconstructed.num_points,
        num_matched_voxels=int(np.sum(covered)),
    )


# ============================================================
# Simulation Readiness
# ============================================================


def simulation_readiness(
    export: MujocoExport,
    stability_steps: int = 1000,
    max_penetration: float = 0.01,
) -> SimulationReadinessResult:
    """Evaluate whether exported scene is ready for MuJoCo simulation."""
    # Check XML parsing
    xml_ok = False
    physics_stable = False
    max_pen = 0.0
    steps_run = 0

    try:
        import mujoco  # type: ignore[import-untyped]

        model = mujoco.MjModel.from_xml_string(export.xml_content)
        xml_ok = True

        data = mujoco.MjData(model)
        for i in range(stability_steps):
            mujoco.mj_step(model, data)
            steps_run = i + 1
            # Check for NaN/Inf
            if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
                break

        max_pen = float(np.max(np.abs(data.contact.dist))) if data.ncon > 0 else 0.0
        physics_stable = (
            np.isfinite(data.qpos).all()
            and np.isfinite(data.qvel).all()
            and max_pen < max_penetration
        )

    except ImportError:
        logger.warning("MuJoCo not installed, skipping physics stability check")
    except Exception as e:
        logger.warning("MuJoCo simulation check failed: %s", e)

    # Check collision meshes
    collision_valid, _ = validate_collision_meshes(export.collision_mesh_paths)
    total_collision_faces = 0
    for p in export.collision_mesh_paths:
        if p.exists():
            import trimesh  # type: ignore[import-untyped]

            m = trimesh.load(str(p))
            if hasattr(m, "faces"):
                total_collision_faces += len(m.faces)

    return SimulationReadinessResult(
        collision_geometry_valid=collision_valid,
        num_collision_bodies=len(export.collision_mesh_paths),
        total_collision_faces=total_collision_faces,
        physics_stable=physics_stable,
        max_penetration_depth=max_pen,
        simulation_steps_run=steps_run,
        xml_parses_ok=xml_ok,
    )


def validate_collision_meshes(
    mesh_paths: list[Path],
) -> tuple[bool, list[str]]:
    """Check that all collision meshes are valid for physics simulation."""
    errors: list[str] = []
    for path in mesh_paths:
        if not path.exists():
            errors.append(f"Mesh not found: {path}")
            continue
        try:
            import trimesh  # type: ignore[import-untyped]

            mesh = trimesh.load(str(path))
            if not hasattr(mesh, "faces") or len(mesh.faces) == 0:
                errors.append(f"Empty mesh: {path}")
            elif not mesh.is_volume:
                errors.append(f"Non-volume mesh (not watertight): {path}")
        except Exception as e:
            errors.append(f"Failed to load {path}: {e}")

    return len(errors) == 0, errors

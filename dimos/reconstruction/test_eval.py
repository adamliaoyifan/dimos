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

"""Tests for reconstruction evaluation metrics."""

from __future__ import annotations

import numpy as np
import pytest

from dimos.reconstruction.config import EvalConfig
from dimos.reconstruction.eval.evaluator import Evaluator, PipelineEvaluation
from dimos.reconstruction.eval.metrics import (
    alignment_residual,
    chamfer_distance,
    completeness,
    geometric_accuracy,
    hausdorff_distance,
    mesh_quality,
    point_cloud_rmse,
)
from dimos.reconstruction.types import PointCloudData, Sim3Transform, TriangleMesh


# ============================================================
# Chamfer distance
# ============================================================


class TestChamferDistance:
    def test_identical_clouds_yield_zero(
        self, sample_point_cloud: PointCloudData
    ) -> None:
        mean, std = chamfer_distance(sample_point_cloud, sample_point_cloud)
        assert mean == pytest.approx(0.0, abs=1e-10)
        assert std == pytest.approx(0.0, abs=1e-10)

    def test_noisy_cloud_yields_small_distance(
        self, sample_point_cloud: PointCloudData, noisy_point_cloud: PointCloudData
    ) -> None:
        mean, std = chamfer_distance(sample_point_cloud, noisy_point_cloud)
        # Noise sigma=0.01, so Chamfer should be on that order
        assert mean < 0.05
        assert mean > 0.0

    def test_distant_clouds_yield_large_distance(
        self, sample_point_cloud: PointCloudData
    ) -> None:
        shifted = PointCloudData(points=sample_point_cloud.points + 100.0)
        mean, _ = chamfer_distance(sample_point_cloud, shifted, num_samples=500)
        assert mean > 90.0


# ============================================================
# Hausdorff distance
# ============================================================


class TestHausdorffDistance:
    def test_identical_clouds_yield_zero(
        self, sample_point_cloud: PointCloudData
    ) -> None:
        d = hausdorff_distance(sample_point_cloud, sample_point_cloud)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_percentile_leq_full(
        self, sample_point_cloud: PointCloudData, noisy_point_cloud: PointCloudData
    ) -> None:
        h100 = hausdorff_distance(sample_point_cloud, noisy_point_cloud)
        h95 = hausdorff_distance(
            sample_point_cloud, noisy_point_cloud, percentile=95.0
        )
        assert h95 <= h100


# ============================================================
# RMSE
# ============================================================


class TestPointCloudRMSE:
    def test_identical_clouds_yield_zero(
        self, sample_point_cloud: PointCloudData
    ) -> None:
        r = point_cloud_rmse(sample_point_cloud, sample_point_cloud)
        assert r == pytest.approx(0.0, abs=1e-10)

    def test_known_offset(self) -> None:
        pts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        shifted = pts + np.array([0.1, 0, 0])
        r = point_cloud_rmse(PointCloudData(points=pts), PointCloudData(points=shifted))
        assert r == pytest.approx(0.1, abs=1e-6)


# ============================================================
# Geometric accuracy (aggregate)
# ============================================================


class TestGeometricAccuracy:
    def test_returns_all_fields(
        self, sample_point_cloud: PointCloudData
    ) -> None:
        result = geometric_accuracy(
            sample_point_cloud, sample_point_cloud, num_chamfer_samples=500
        )
        assert result.chamfer_distance == pytest.approx(0.0, abs=1e-10)
        assert result.rmse == pytest.approx(0.0, abs=1e-10)
        assert result.hausdorff_distance == pytest.approx(0.0, abs=1e-10)


# ============================================================
# Alignment residual
# ============================================================


class TestAlignmentResidual:
    def test_identity_transform_zero_residual(
        self, sample_point_cloud: PointCloudData
    ) -> None:
        identity = Sim3Transform.identity()
        result = alignment_residual(
            sample_point_cloud.points, sample_point_cloud.points, identity
        )
        assert result.residual_rmse == pytest.approx(0.0, abs=1e-10)
        assert result.inlier_ratio == pytest.approx(1.0)

    def test_known_transform_recoverable(
        self,
        sample_point_cloud: PointCloudData,
        transformed_point_cloud: PointCloudData,
        known_transform: Sim3Transform,
    ) -> None:
        result = alignment_residual(
            sample_point_cloud.points,
            transformed_point_cloud.points,
            known_transform,
            inlier_threshold=0.001,
        )
        assert result.residual_rmse < 1e-10
        assert result.scale_factor == pytest.approx(2.0)
        assert result.inlier_ratio == pytest.approx(1.0)


# ============================================================
# Mesh quality
# ============================================================


class TestMeshQuality:
    def test_cube_mesh(self, sample_mesh: TriangleMesh) -> None:
        result = mesh_quality(sample_mesh)
        assert result.num_vertices == 8
        assert result.num_faces == 12
        assert result.is_watertight is True
        assert result.num_degenerate_faces == 0
        assert result.surface_area > 0

    def test_degenerate_face_detected(self) -> None:
        vertices = np.array(
            [[0, 0, 0], [1, 0, 0], [0.5, 0, 0]],  # collinear
            dtype=np.float64,
        )
        faces = np.array([[0, 1, 2]], dtype=np.int64)
        mesh = TriangleMesh(vertices=vertices, faces=faces)
        result = mesh_quality(mesh)
        assert result.num_degenerate_faces == 1


# ============================================================
# Completeness
# ============================================================


class TestCompleteness:
    def test_identical_clouds_perfect_score(
        self, sample_point_cloud: PointCloudData
    ) -> None:
        result = completeness(sample_point_cloud, sample_point_cloud, voxel_size=0.1)
        assert result.precision == pytest.approx(1.0)
        assert result.recall == pytest.approx(1.0)
        assert result.f_score == pytest.approx(1.0)

    def test_no_overlap_yields_zero(self, sample_point_cloud: PointCloudData) -> None:
        far_away = PointCloudData(points=sample_point_cloud.points + 100.0)
        result = completeness(far_away, sample_point_cloud, voxel_size=0.1)
        assert result.f_score == pytest.approx(0.0)

    def test_distance_based_mode(self, sample_point_cloud: PointCloudData) -> None:
        result = completeness(
            sample_point_cloud,
            sample_point_cloud,
            distance_threshold=0.1,
        )
        assert result.recall == pytest.approx(1.0)


# ============================================================
# Evaluator cross-run comparison
# ============================================================


class TestEvaluatorComparison:
    def test_compare_two_runs(self, sample_point_cloud: PointCloudData) -> None:
        config = EvalConfig(chamfer_num_samples=500, hausdorff_num_samples=500)
        evaluator = Evaluator(config)

        from dimos.reconstruction.types import FusedScene

        scene = FusedScene(point_cloud=sample_point_cloud)

        eval1 = evaluator.evaluate(
            fused_scene=scene,
            ground_truth_pcd=sample_point_cloud,
            run_name="run_a",
        )
        eval2 = evaluator.evaluate(
            fused_scene=scene,
            ground_truth_pcd=sample_point_cloud,
            run_name="run_b",
        )

        table = Evaluator.compare_runs([eval1, eval2])
        assert len(table.run_names) == 2
        assert "chamfer" in table.metrics
        # Same data → same metrics
        for values in table.metrics.values():
            assert values[0] == pytest.approx(values[1], abs=1e-10)

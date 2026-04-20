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

"""Integration tests for the reconstruction pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from dimos.reconstruction.config import EvalConfig, PipelineConfig
from dimos.reconstruction.eval.evaluator import Evaluator
from dimos.reconstruction.eval.report import print_evaluation_report
from dimos.reconstruction.types import (
    FusedScene,
    PointCloudData,
    ReconstructionResult,
    Sim3Transform,
    TriangleMesh,
)


class TestPipelineEvaluation:
    """Test the evaluation pipeline on synthetic data (no external tools needed)."""

    @pytest.fixture
    def cube_scene(self) -> tuple[PointCloudData, TriangleMesh]:
        """Generate a synthetic cube room scene."""
        rng = np.random.default_rng(123)

        # Points on cube surfaces
        n_per_face = 500
        points_list = []

        for axis in range(3):
            for val in [0.0, 5.0]:
                pts = rng.uniform(0, 5, size=(n_per_face, 3))
                pts[:, axis] = val
                points_list.append(pts)

        points = np.concatenate(points_list, axis=0)
        pcd = PointCloudData(points=points)

        # Simple cube mesh
        vertices = np.array(
            [
                [0, 0, 0], [5, 0, 0], [5, 5, 0], [0, 5, 0],
                [0, 0, 5], [5, 0, 5], [5, 5, 5], [0, 5, 5],
            ],
            dtype=np.float64,
        )
        faces = np.array(
            [
                [0, 2, 1], [0, 3, 2],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [0, 4, 7], [0, 7, 3],
                [1, 2, 6], [1, 6, 5],
            ],
            dtype=np.int64,
        )
        mesh = TriangleMesh(vertices=vertices, faces=faces)

        return pcd, mesh

    def test_full_evaluation_on_synthetic(
        self, cube_scene: tuple[PointCloudData, TriangleMesh]
    ) -> None:
        """Run full evaluation pipeline on synthetic cube scene."""
        pcd, mesh = cube_scene
        config = EvalConfig(
            chamfer_num_samples=1000,
            hausdorff_num_samples=1000,
            completeness_voxel_size=0.5,
        )
        evaluator = Evaluator(config)

        scene = FusedScene(point_cloud=pcd, mesh=mesh)

        evaluation = evaluator.evaluate(
            fused_scene=scene,
            ground_truth_pcd=pcd,  # self-comparison
            run_name="synthetic_test",
        )

        # Self-comparison should yield perfect scores
        assert evaluation.geometric_accuracy is not None
        assert evaluation.geometric_accuracy.chamfer_distance == pytest.approx(
            0.0, abs=1e-6
        )

        assert evaluation.completeness_result is not None
        assert evaluation.completeness_result.f_score == pytest.approx(1.0)

        assert evaluation.mesh_quality_result is not None
        assert evaluation.mesh_quality_result.is_watertight is True

        # Print report (verifies no crash)
        print_evaluation_report(evaluation)

    def test_cross_run_comparison(
        self, cube_scene: tuple[PointCloudData, TriangleMesh]
    ) -> None:
        """Compare two evaluation runs."""
        pcd, mesh = cube_scene
        config = EvalConfig(chamfer_num_samples=500, hausdorff_num_samples=500)
        evaluator = Evaluator(config)

        scene = FusedScene(point_cloud=pcd, mesh=mesh)

        eval_a = evaluator.evaluate(
            fused_scene=scene, ground_truth_pcd=pcd, run_name="run_a"
        )

        # Create a noisy version
        rng = np.random.default_rng(99)
        noisy_pcd = PointCloudData(points=pcd.points + rng.normal(0, 0.1, pcd.points.shape))
        noisy_scene = FusedScene(point_cloud=noisy_pcd, mesh=mesh)

        eval_b = evaluator.evaluate(
            fused_scene=noisy_scene, ground_truth_pcd=pcd, run_name="run_b"
        )

        table = Evaluator.compare_runs([eval_a, eval_b])
        assert len(table.run_names) == 2

        # Run A (perfect) should have better chamfer than run B (noisy)
        chamfer_values = table.metrics.get("chamfer", [])
        assert chamfer_values[0] is not None and chamfer_values[1] is not None
        assert chamfer_values[0] < chamfer_values[1]

        # Print table (verifies no crash)
        table.print_table()

    def test_per_stage_evaluation(
        self, cube_scene: tuple[PointCloudData, TriangleMesh]
    ) -> None:
        """Evaluate individual reconstruction stages."""
        pcd, mesh = cube_scene
        config = EvalConfig(chamfer_num_samples=500, hausdorff_num_samples=500)
        evaluator = Evaluator(config)

        recon = ReconstructionResult(
            point_cloud=pcd, mesh=mesh, source="test"
        )

        stage_eval = evaluator.evaluate_stage("lidar_recon", recon, pcd)
        assert "geometric_accuracy" in stage_eval.metrics
        assert "mesh_quality" in stage_eval.metrics

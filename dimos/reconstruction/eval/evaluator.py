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

"""Evaluation runner: orchestrates metric computation across pipeline outputs."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dimos.reconstruction.config import EvalConfig
from dimos.reconstruction.eval.metrics import (
    AlignmentQualityResult,
    CompletenessResult,
    GeometricAccuracyResult,
    MeshQualityResult,
    SimulationReadinessResult,
    alignment_quality_from_clouds,
    completeness,
    geometric_accuracy,
    mesh_quality,
    simulation_readiness,
)
from dimos.reconstruction.types import (
    AlignmentResult,
    FusedScene,
    MujocoExport,
    PointCloudData,
    ReconstructionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class StageEvaluation:
    """Evaluation results for a single pipeline stage."""

    stage_name: str
    metrics: dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineEvaluation:
    """Full evaluation of a pipeline run."""

    run_name: str
    stages: list[StageEvaluation] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

    geometric_accuracy: GeometricAccuracyResult | None = None
    alignment_quality: AlignmentQualityResult | None = None
    mesh_quality_result: MeshQualityResult | None = None
    completeness_result: CompletenessResult | None = None
    simulation_readiness_result: SimulationReadinessResult | None = None

    def summary_dict(self) -> dict[str, float]:
        """Flat dictionary of key scalar metrics for cross-run comparison."""
        d: dict[str, float] = {}
        if self.geometric_accuracy:
            d["chamfer"] = self.geometric_accuracy.chamfer_distance
            d["hausdorff_95"] = self.geometric_accuracy.hausdorff_95
            d["rmse"] = self.geometric_accuracy.rmse
        if self.alignment_quality:
            d["align_rmse"] = self.alignment_quality.residual_rmse
            d["align_inlier_ratio"] = self.alignment_quality.inlier_ratio
        if self.mesh_quality_result:
            d["mesh_faces"] = float(self.mesh_quality_result.num_faces)
            d["mesh_watertight"] = float(self.mesh_quality_result.is_watertight)
            d["mesh_normal_consistency"] = self.mesh_quality_result.normal_consistency
        if self.completeness_result:
            d["completeness_f_score"] = self.completeness_result.f_score
            d["completeness_recall"] = self.completeness_result.recall
            d["completeness_precision"] = self.completeness_result.precision
        if self.simulation_readiness_result:
            d["sim_stable"] = float(self.simulation_readiness_result.physics_stable)
            d["sim_xml_ok"] = float(self.simulation_readiness_result.xml_parses_ok)
        return d

    def __str__(self) -> str:
        lines = [f"=== Pipeline Evaluation: {self.run_name} ==="]
        for k, v in self.summary_dict().items():
            lines.append(f"  {k}: {v:.6f}")
        return "\n".join(lines)


@dataclass
class ComparisonTable:
    """Side-by-side comparison of multiple pipeline evaluations."""

    run_names: list[str]
    metrics: dict[str, list[float | None]]

    def print_table(self) -> None:
        """Print comparison table."""
        # Header
        header = f"{'Metric':<30}" + "".join(f"{n:>15}" for n in self.run_names)
        print(header)
        print("-" * len(header))

        for metric_name, values in sorted(self.metrics.items()):
            row = f"{metric_name:<30}"
            for v in values:
                if v is None:
                    row += f"{'N/A':>15}"
                else:
                    row += f"{v:>15.6f}"
            print(row)

    def to_csv(self, path: str) -> None:
        """Export comparison to CSV."""
        import csv

        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["metric"] + self.run_names)
            for metric_name, values in sorted(self.metrics.items()):
                writer.writerow(
                    [metric_name] + [f"{v:.6f}" if v is not None else "" for v in values]
                )


class Evaluator:
    """Runs all applicable evaluations given pipeline outputs.

    Usage:
        evaluator = Evaluator(config)
        result = evaluator.evaluate(
            fused_scene=scene,
            export=mujoco_export,
            ground_truth_pcd=gt_cloud,
        )
        print(result)
    """

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def evaluate(
        self,
        fused_scene: FusedScene | None = None,
        export: MujocoExport | None = None,
        ground_truth_pcd: PointCloudData | None = None,
        alignment_result: AlignmentResult | None = None,
        per_stage_reconstructions: dict[str, ReconstructionResult] | None = None,
        run_name: str = "default",
    ) -> PipelineEvaluation:
        """Run all applicable evaluations. Skips metrics needing unavailable data."""
        evaluation = PipelineEvaluation(run_name=run_name)

        # Geometric accuracy (requires ground truth)
        if ground_truth_pcd and fused_scene:
            logger.info("Computing geometric accuracy...")
            evaluation.geometric_accuracy = geometric_accuracy(
                source=fused_scene.point_cloud,
                target=ground_truth_pcd,
                num_chamfer_samples=self.config.chamfer_num_samples,
                num_hausdorff_samples=self.config.hausdorff_num_samples,
            )
            evaluation.stages.append(
                StageEvaluation(
                    stage_name="geometric_accuracy",
                    metrics={"result": evaluation.geometric_accuracy},
                )
            )

        # Alignment quality
        if alignment_result and fused_scene:
            logger.info("Computing alignment quality...")
            # Use the aligned reconstruction vs the target (LiDAR)
            target_pcd = None
            if fused_scene.sources:
                for src in fused_scene.sources:
                    if src.is_metric:
                        target_pcd = src.point_cloud
                        break
            if target_pcd:
                evaluation.alignment_quality = alignment_quality_from_clouds(
                    aligned_source=alignment_result.aligned_reconstruction.point_cloud,
                    target=target_pcd,
                )
                evaluation.stages.append(
                    StageEvaluation(
                        stage_name="alignment_quality",
                        metrics={"result": evaluation.alignment_quality},
                    )
                )

        # Mesh quality (no ground truth needed)
        if fused_scene and fused_scene.mesh:
            logger.info("Computing mesh quality...")
            evaluation.mesh_quality_result = mesh_quality(fused_scene.mesh)
            evaluation.stages.append(
                StageEvaluation(
                    stage_name="mesh_quality",
                    metrics={"result": evaluation.mesh_quality_result},
                )
            )

        # Completeness (requires ground truth)
        if ground_truth_pcd and fused_scene:
            logger.info("Computing completeness...")
            evaluation.completeness_result = completeness(
                reconstructed=fused_scene.point_cloud,
                ground_truth=ground_truth_pcd,
                voxel_size=self.config.completeness_voxel_size,
            )
            evaluation.stages.append(
                StageEvaluation(
                    stage_name="completeness",
                    metrics={"result": evaluation.completeness_result},
                )
            )

        # Simulation readiness
        if export:
            logger.info("Computing simulation readiness...")
            evaluation.simulation_readiness_result = simulation_readiness(
                export=export,
                stability_steps=self.config.mujoco_stability_steps,
                max_penetration=self.config.mujoco_stability_max_penetration,
            )
            evaluation.stages.append(
                StageEvaluation(
                    stage_name="simulation_readiness",
                    metrics={"result": evaluation.simulation_readiness_result},
                )
            )

        # Per-stage evaluations against ground truth
        if per_stage_reconstructions and ground_truth_pcd:
            for stage_name, recon in per_stage_reconstructions.items():
                stage_eval = self.evaluate_stage(stage_name, recon, ground_truth_pcd)
                evaluation.stages.append(stage_eval)

        return evaluation

    def evaluate_stage(
        self,
        stage_name: str,
        reconstruction: ReconstructionResult,
        ground_truth: PointCloudData | None = None,
    ) -> StageEvaluation:
        """Evaluate a single stage's output."""
        metrics: dict[str, Any] = {}

        if ground_truth:
            metrics["geometric_accuracy"] = geometric_accuracy(
                source=reconstruction.point_cloud,
                target=ground_truth,
                num_chamfer_samples=self.config.chamfer_num_samples,
                num_hausdorff_samples=self.config.hausdorff_num_samples,
            )

        if reconstruction.mesh:
            metrics["mesh_quality"] = mesh_quality(reconstruction.mesh)

        return StageEvaluation(stage_name=stage_name, metrics=metrics)

    @staticmethod
    def compare_runs(evaluations: list[PipelineEvaluation]) -> ComparisonTable:
        """Compare multiple pipeline runs side by side."""
        run_names = [e.run_name for e in evaluations]

        # Collect all metric keys
        all_keys: set[str] = set()
        summaries = [e.summary_dict() for e in evaluations]
        for s in summaries:
            all_keys.update(s.keys())

        metrics: dict[str, list[float | None]] = {}
        for key in sorted(all_keys):
            metrics[key] = [s.get(key) for s in summaries]

        return ComparisonTable(run_names=run_names, metrics=metrics)

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

"""Report generation for reconstruction evaluation results."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from dimos.reconstruction.eval.evaluator import ComparisonTable, PipelineEvaluation


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj: object) -> object:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def print_evaluation_report(evaluation: PipelineEvaluation) -> None:
    """Print a formatted evaluation report to stdout."""
    print(f"\n{'=' * 60}")
    print(f"  Reconstruction Evaluation: {evaluation.run_name}")
    print(f"  Timestamp: {evaluation.timestamp}")
    print(f"{'=' * 60}\n")

    if evaluation.geometric_accuracy:
        ga = evaluation.geometric_accuracy
        print("  Geometric Accuracy:")
        print(f"    Chamfer distance:  {ga.chamfer_distance:.6f} (std: {ga.chamfer_distance_std:.6f})")
        print(f"    Hausdorff (100%):  {ga.hausdorff_distance:.6f}")
        print(f"    Hausdorff (95%):   {ga.hausdorff_95:.6f}")
        print(f"    RMSE:              {ga.rmse:.6f}")
        print(f"    Samples (src/tgt): {ga.num_samples_source} / {ga.num_samples_target}")
        print()

    if evaluation.alignment_quality:
        aq = evaluation.alignment_quality
        print("  Alignment Quality:")
        print(f"    Residual RMSE:     {aq.residual_rmse:.6f}")
        print(f"    Residual max:      {aq.residual_max:.6f}")
        print(f"    Scale factor:      {aq.scale_factor:.6f}")
        print(f"    Scale consistency: {aq.scale_consistency:.6f}")
        print(f"    Inlier ratio:      {aq.inlier_ratio:.4f}")
        print(f"    Correspondences:   {aq.num_correspondences}")
        print()

    if evaluation.mesh_quality_result:
        mq = evaluation.mesh_quality_result
        print("  Mesh Quality:")
        print(f"    Vertices:          {mq.num_vertices:,}")
        print(f"    Faces:             {mq.num_faces:,}")
        print(f"    Watertight:        {mq.is_watertight}")
        print(f"    Non-manifold:      {mq.num_non_manifold_edges}")
        print(f"    Degenerate faces:  {mq.num_degenerate_faces}")
        print(f"    Normal consistency:{mq.normal_consistency:.4f}")
        print(f"    Bounding box:      {mq.bounding_box_extent}")
        print(f"    Surface area:      {mq.surface_area:.4f} m²")
        print()

    if evaluation.completeness_result:
        cr = evaluation.completeness_result
        print("  Completeness:")
        print(f"    Precision:         {cr.precision:.4f}")
        print(f"    Recall:            {cr.recall:.4f}")
        print(f"    F-score:           {cr.f_score:.4f}")
        print(f"    Voxel size:        {cr.voxel_size_used:.4f}")
        print(f"    GT / Recon / Match:{cr.num_gt_voxels} / {cr.num_recon_voxels} / {cr.num_matched_voxels}")
        print()

    if evaluation.simulation_readiness_result:
        sr = evaluation.simulation_readiness_result
        print("  Simulation Readiness:")
        print(f"    XML parses:        {sr.xml_parses_ok}")
        print(f"    Collision valid:   {sr.collision_geometry_valid}")
        print(f"    Collision bodies:  {sr.num_collision_bodies}")
        print(f"    Physics stable:    {sr.physics_stable}")
        print(f"    Max penetration:   {sr.max_penetration_depth:.6f}")
        print(f"    Steps run:         {sr.simulation_steps_run}")
        print()

    print(f"{'=' * 60}\n")


def save_evaluation_json(evaluation: PipelineEvaluation, path: Path) -> None:
    """Serialize evaluation to JSON for programmatic comparison."""
    data = {
        "run_name": evaluation.run_name,
        "timestamp": evaluation.timestamp,
        "summary": evaluation.summary_dict(),
        "stages": [
            {
                "stage_name": s.stage_name,
                "timestamp": s.timestamp,
                "metrics": _serialize_metrics(s.metrics),
            }
            for s in evaluation.stages
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2, cls=_NumpyEncoder)


def save_comparison_csv(comparison: ComparisonTable, path: Path) -> None:
    """Save run comparison to CSV."""
    comparison.to_csv(str(path))


def _serialize_metrics(metrics: dict) -> dict:
    """Recursively convert metric dataclasses to dicts."""
    result = {}
    for k, v in metrics.items():
        if hasattr(v, "__dataclass_fields__"):
            result[k] = asdict(v)
        else:
            result[k] = v
    return result

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

"""Pipeline orchestrator: compose stages, run end-to-end, evaluate."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from dimos.reconstruction.config import PipelineConfig
from dimos.reconstruction.eval.evaluator import Evaluator, PipelineEvaluation
from dimos.reconstruction.stages.alignment import AlignmentInput, AlignmentStage
from dimos.reconstruction.stages.export import ExportStage
from dimos.reconstruction.stages.fusion import FusionInput, FusionStage
from dimos.reconstruction.stages.image_recon import ImageReconStage
from dimos.reconstruction.stages.ingest import IngestInput, IngestStage
from dimos.reconstruction.stages.lidar_recon import LidarReconStage
from dimos.reconstruction.types import (
    AlignmentResult,
    FusedScene,
    MujocoExport,
    PointCloudData,
    ReconstructionResult,
)

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Complete result of a pipeline run."""

    config: PipelineConfig
    ingest_output: Any | None = None
    image_recon: ReconstructionResult | None = None
    lidar_recon: ReconstructionResult | None = None
    alignment: AlignmentResult | None = None
    fused_scene: FusedScene | None = None
    export: MujocoExport | None = None
    evaluation: PipelineEvaluation | None = None
    stage_timings: dict[str, float] = field(default_factory=dict)


class ReconstructionPipeline:
    """Orchestrates the full reconstruction pipeline.

    Usage:
        config = PipelineConfig(...)
        pipeline = ReconstructionPipeline(config)
        result = pipeline.run()
        pipeline.evaluate(result)

    Swap algorithms:
        pipeline.set_stage("image_recon", MyCustomStage(my_config))
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self._stages: dict[str, Any] = {
            "ingest": IngestStage(config.ingest),
            "image_recon": ImageReconStage(config.image_recon),
            "lidar_recon": LidarReconStage(config.lidar_recon),
            "alignment": AlignmentStage(config.alignment),
            "fusion": FusionStage(config.fusion),
            "export": ExportStage(config.export),
        }
        self._evaluator = Evaluator(config.eval)

    def set_stage(self, name: str, stage: Any) -> None:
        """Replace a pipeline stage for algorithm swapping."""
        if name not in self._stages:
            raise ValueError(
                f"Unknown stage '{name}'. Valid: {list(self._stages.keys())}"
            )
        self._stages[name] = stage

    def run(
        self,
        ground_truth_pcd: PointCloudData | None = None,
    ) -> PipelineResult:
        """Run the full pipeline end-to-end."""
        result = PipelineResult(config=self.config)

        # Stage 1: Ingest
        ingest_output = self._run_timed(
            "ingest", self._stages["ingest"], IngestInput(), result
        )
        result.ingest_output = ingest_output

        reconstructions: list[ReconstructionResult] = []
        per_stage: dict[str, ReconstructionResult] = {}

        # Stage 2: Image reconstruction
        if not self.config.skip_image_recon and ingest_output.image_dataset:
            image_recon = self._run_timed(
                "image_recon",
                self._stages["image_recon"],
                ingest_output.image_dataset,
                result,
            )
            result.image_recon = image_recon
            per_stage["image_recon"] = image_recon

        # Stage 3: LiDAR reconstruction
        if not self.config.skip_lidar_recon and ingest_output.lidar_point_cloud:
            lidar_recon = self._run_timed(
                "lidar_recon",
                self._stages["lidar_recon"],
                ingest_output.lidar_point_cloud,
                result,
            )
            result.lidar_recon = lidar_recon
            reconstructions.append(lidar_recon)
            per_stage["lidar_recon"] = lidar_recon

        # Stage 4: Alignment (if we have both)
        if (
            not self.config.skip_alignment
            and result.image_recon
            and result.lidar_recon
        ):
            alignment_input = AlignmentInput(
                source=result.image_recon,
                target=result.lidar_recon,
            )
            alignment_result = self._run_timed(
                "alignment",
                self._stages["alignment"],
                alignment_input,
                result,
            )
            result.alignment = alignment_result
            reconstructions.append(alignment_result.aligned_reconstruction)
        elif result.image_recon and not result.lidar_recon:
            # Only image reconstruction, no alignment needed
            reconstructions.append(result.image_recon)

        # Stage 5: Fusion
        if reconstructions:
            fusion_input = FusionInput(reconstructions=reconstructions)
            fused = self._run_timed(
                "fusion", self._stages["fusion"], fusion_input, result
            )
            result.fused_scene = fused

            # Stage 6: Export
            export = self._run_timed(
                "export", self._stages["export"], fused, result
            )
            result.export = export

        # Evaluate
        if ground_truth_pcd or result.fused_scene:
            result.evaluation = self.evaluate(result, ground_truth_pcd)

        return result

    def run_stage(self, stage_name: str, input_data: Any) -> Any:
        """Run a single stage by name."""
        stage = self._stages.get(stage_name)
        if stage is None:
            raise ValueError(f"Unknown stage: {stage_name}")
        return stage.run(input_data)

    def evaluate(
        self,
        result: PipelineResult,
        ground_truth_pcd: PointCloudData | None = None,
    ) -> PipelineEvaluation:
        """Evaluate a pipeline result."""
        per_stage = {}
        if result.image_recon:
            per_stage["image_recon"] = result.image_recon
        if result.lidar_recon:
            per_stage["lidar_recon"] = result.lidar_recon

        return self._evaluator.evaluate(
            fused_scene=result.fused_scene,
            export=result.export,
            ground_truth_pcd=ground_truth_pcd,
            alignment_result=result.alignment,
            per_stage_reconstructions=per_stage if ground_truth_pcd else None,
            run_name=self.config.run_name,
        )

    def _run_timed(
        self, name: str, stage: Any, input_data: Any, result: PipelineResult
    ) -> Any:
        """Run a stage with timing."""
        logger.info("Running stage: %s", name)
        t0 = time.monotonic()

        errors = stage.validate_input(input_data)
        if errors:
            raise ValueError(f"Stage '{name}' validation failed: {errors}")

        output = stage.run(input_data)
        elapsed = time.monotonic() - t0
        result.stage_timings[name] = elapsed
        logger.info("Stage '%s' completed in %.2fs", name, elapsed)
        return output

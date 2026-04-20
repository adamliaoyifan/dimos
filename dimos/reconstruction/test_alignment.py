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

"""Tests for alignment algorithms (Umeyama, ICP, transform application)."""

from __future__ import annotations

import numpy as np
import pytest

from dimos.reconstruction.stages.alignment import (
    apply_transform_to_reconstruction,
    umeyama_sim3,
)
from dimos.reconstruction.types import (
    PointCloudData,
    ReconstructionResult,
    Sim3Transform,
)


class TestUmeyamaSim3:
    def test_identity_recovery(self, sample_point_cloud: PointCloudData) -> None:
        """Identical source and target should recover identity."""
        transform = umeyama_sim3(
            sample_point_cloud.points, sample_point_cloud.points
        )
        assert transform.scale == pytest.approx(1.0, abs=1e-6)
        np.testing.assert_allclose(transform.rotation, np.eye(3), atol=1e-6)
        np.testing.assert_allclose(transform.translation, np.zeros(3), atol=1e-6)

    def test_known_transform_recovery(
        self,
        sample_point_cloud: PointCloudData,
        transformed_point_cloud: PointCloudData,
        known_transform: Sim3Transform,
    ) -> None:
        """Recover a known Sim(3) transform."""
        recovered = umeyama_sim3(
            sample_point_cloud.points, transformed_point_cloud.points
        )

        assert recovered.scale == pytest.approx(known_transform.scale, rel=1e-6)
        np.testing.assert_allclose(
            recovered.rotation, known_transform.rotation, atol=1e-6
        )
        np.testing.assert_allclose(
            recovered.translation, known_transform.translation, atol=1e-6
        )

    def test_residual_after_recovery(
        self,
        sample_point_cloud: PointCloudData,
        transformed_point_cloud: PointCloudData,
    ) -> None:
        """After applying recovered transform, residual should be ~0."""
        recovered = umeyama_sim3(
            sample_point_cloud.points, transformed_point_cloud.points
        )
        aligned = recovered.apply(sample_point_cloud.points)
        residual = np.linalg.norm(aligned - transformed_point_cloud.points, axis=1)
        assert np.max(residual) < 1e-8

    def test_pure_scale(self, rng: np.random.Generator) -> None:
        """Recover pure scale (no rotation, no translation)."""
        src = rng.uniform(0, 1, size=(500, 3))
        scale = 3.5
        tgt = src * scale

        transform = umeyama_sim3(src, tgt)
        assert transform.scale == pytest.approx(scale, rel=1e-6)

    def test_pure_translation(self, rng: np.random.Generator) -> None:
        """Recover pure translation (no rotation, no scale)."""
        src = rng.uniform(0, 1, size=(500, 3))
        t = np.array([10.0, -5.0, 3.0])
        tgt = src + t

        transform = umeyama_sim3(src, tgt)
        assert transform.scale == pytest.approx(1.0, rel=1e-6)
        np.testing.assert_allclose(transform.translation, t, atol=1e-6)

    def test_minimum_points(self) -> None:
        """Should work with exactly 3 points."""
        src = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
        tgt = src * 2.0 + np.array([1, 1, 1])

        transform = umeyama_sim3(src, tgt)
        aligned = transform.apply(src)
        np.testing.assert_allclose(aligned, tgt, atol=1e-8)


class TestApplyTransform:
    def test_identity_preserves(self, sample_point_cloud: PointCloudData) -> None:
        recon = ReconstructionResult(
            point_cloud=sample_point_cloud, source="test"
        )
        identity = Sim3Transform.identity()
        result = apply_transform_to_reconstruction(recon, identity)
        np.testing.assert_allclose(
            result.point_cloud.points, sample_point_cloud.points, atol=1e-10
        )

    def test_transform_sets_metric(
        self, sample_point_cloud: PointCloudData
    ) -> None:
        recon = ReconstructionResult(
            point_cloud=sample_point_cloud, is_metric=False, source="test"
        )
        result = apply_transform_to_reconstruction(
            recon, Sim3Transform.identity()
        )
        assert result.is_metric is True

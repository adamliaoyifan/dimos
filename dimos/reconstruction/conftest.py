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

"""Shared test fixtures for reconstruction pipeline tests."""

from __future__ import annotations

import numpy as np
import pytest

from dimos.reconstruction.types import PointCloudData, Sim3Transform, TriangleMesh


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded RNG for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def sample_point_cloud(rng: np.random.Generator) -> PointCloudData:
    """Unit cube point cloud with 1000 random points in [0, 1]^3."""
    points = rng.uniform(0, 1, size=(1000, 3))
    colors = rng.uniform(0, 1, size=(1000, 3))
    return PointCloudData(points=points, colors=colors)


@pytest.fixture
def sample_mesh() -> TriangleMesh:
    """Simple watertight cube mesh (8 vertices, 12 faces)."""
    vertices = np.array(
        [
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],  # bottom
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],  # top
        ],
        dtype=np.float64,
    )
    faces = np.array(
        [
            # bottom
            [0, 2, 1], [0, 3, 2],
            # top
            [4, 5, 6], [4, 6, 7],
            # front
            [0, 1, 5], [0, 5, 4],
            # back
            [2, 3, 7], [2, 7, 6],
            # left
            [0, 4, 7], [0, 7, 3],
            # right
            [1, 2, 6], [1, 6, 5],
        ],
        dtype=np.int64,
    )
    return TriangleMesh(vertices=vertices, faces=faces)


@pytest.fixture
def noisy_point_cloud(
    sample_point_cloud: PointCloudData, rng: np.random.Generator
) -> PointCloudData:
    """Sample point cloud with added Gaussian noise (sigma=0.01)."""
    noise = rng.normal(0, 0.01, size=sample_point_cloud.points.shape)
    return PointCloudData(
        points=sample_point_cloud.points + noise,
        colors=sample_point_cloud.colors,
    )


@pytest.fixture
def known_transform() -> Sim3Transform:
    """A known Sim(3) transform: scale=2.0, 90-deg rotation around Z, translate [1,2,3]."""
    theta = np.pi / 2
    R = np.array(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ]
    )
    return Sim3Transform(rotation=R, translation=np.array([1.0, 2.0, 3.0]), scale=2.0)


@pytest.fixture
def transformed_point_cloud(
    sample_point_cloud: PointCloudData, known_transform: Sim3Transform
) -> PointCloudData:
    """sample_point_cloud transformed by known_transform."""
    new_points = known_transform.apply(sample_point_cloud.points)
    return PointCloudData(points=new_points, colors=sample_point_cloud.colors)

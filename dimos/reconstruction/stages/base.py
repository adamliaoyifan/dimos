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

"""Abstract stage interface using Protocol for structural subtyping.

Any class with a matching ``run`` method qualifies as a Stage — no
inheritance required. This follows the PointCloudAccumulator pattern
from ``dimos.mapping.pointclouds.accumulators.protocol``.
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

InputT = TypeVar("InputT", contravariant=True)
OutputT = TypeVar("OutputT", covariant=True)


@runtime_checkable
class Stage(Protocol[InputT, OutputT]):
    """A pipeline stage that transforms InputT → OutputT."""

    @property
    def name(self) -> str:
        """Human-readable stage name for logging and reports."""
        ...

    def run(self, input_data: InputT) -> OutputT:
        """Execute the stage."""
        ...

    def validate_input(self, input_data: InputT) -> list[str]:
        """Return list of validation errors. Empty list means valid."""
        ...

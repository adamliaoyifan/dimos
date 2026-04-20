# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DimOS-specific adapter that bridges DimOS module streams into ros2_visualization."""

from dimos.ros2_visualization.adapters.dimos.streams import DimosAdapter

__all__ = ["DimosAdapter"]

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TF frame name constants.

Import from here instead of scattering string literals throughout the codebase.
Override by passing ``frames=`` to :class:`~dimos.ros2_visualization.core.registry.BridgeRegistry`.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class FrameIds:
    """Mutable set of TF frame identifiers used by all bridges."""

    map: str = "map"
    odom: str = "odom"
    base_link: str = "base_link"
    lidar: str = "lidar"
    camera: str = "camera"
    depth_camera: str = "depth_camera"

    # Static offset from base_link to lidar (x, y, z) in metres
    lidar_offset_xyz: tuple[float, float, float] = (0.0, 0.0, 0.3)
    # Static offset from base_link to forward camera
    camera_offset_xyz: tuple[float, float, float] = (0.1, 0.0, 0.15)


# Default instance — importable as a singleton
DEFAULT_FRAMES = FrameIds()

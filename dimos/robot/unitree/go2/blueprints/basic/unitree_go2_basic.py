#!/usr/bin/env python3

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

import platform
from typing import Any

from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE
from dimos.core.blueprints import autoconnect
from dimos.core.global_config import global_config
from dimos.core.transport import LCMTransport, pSHMTransport
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image
from dimos.protocol.pubsub.impl.lcmpubsub import LCM
from dimos.protocol.service.system_configurator.clock_sync import ClockSyncConfigurator
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.web.websocket_vis.websocket_vis_module import WebsocketVisModule

# Use pSHM for all image streams on all platforms.
# LCM UDP multicast is unreliable for large payloads (640x480 ~900KB, 320x240 ~230KB)
# and pSHM gives zero-copy delivery to all in-process consumers (Rerun bridge, NavDP, etc.).
_image_transports: dict[tuple[str, type], Any] = {
    # Go2 head camera (320x240)
    ("color_image", Image): pSHMTransport(
        "color_image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    ),
    ("depth_image", Image): pSHMTransport(
        "depth_image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    ),
    # RealSense / D455i camera (640x480)
    ("realsense_image", Image): pSHMTransport(
        "realsense_image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    ),
    ("realsense_depth", Image): pSHMTransport(
        "realsense_depth", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    ),
    ("realsense_camera_info", CameraInfo): LCMTransport("/realsense_camera_info", CameraInfo),
}

_transports_base = autoconnect().transports(_image_transports)


def _convert_camera_info(camera_info: Any) -> Any:
    # Pinhole goes to child entities; Transform3D goes to the anchor entity.
    # This avoids the Rerun "frame has two parents" conflict.
    rgb = camera_info.to_rerun(
        image_topic="/world/camera_optical/rgb",
        optical_frame="camera_optical",
        camera_entity="/world/camera_optical",
    )
    depth = camera_info.to_rerun(
        image_topic="/world/camera_optical/depth",
        optical_frame="camera_optical",
        camera_entity="/world/camera_optical",
    )
    rgb_list: list[Any] = rgb if isinstance(rgb, list) else []
    depth_list: list[Any] = depth if isinstance(depth, list) else []
    # De-duplicate the Transform3D anchor entry (both calls emit the same one)
    seen: set[tuple[Any, Any]] = set()
    result: list[Any] = []
    for item in rgb_list + depth_list:
        key = (item[0], type(item[1]).__name__)
        if key not in seen:
            seen.add(key)
            result.append(item)
    return result


def _convert_realsense_camera_info(camera_info: Any) -> Any:
    # Same split pattern for RealSense / D455i.
    rgb = camera_info.to_rerun(
        image_topic="/world/realsense_optical/rgb",
        optical_frame="realsense_optical",
        camera_entity="/world/realsense_optical",
    )
    depth = camera_info.to_rerun(
        image_topic="/world/realsense_optical/depth",
        optical_frame="realsense_optical",
        camera_entity="/world/realsense_optical",
    )
    rgb_list: list[Any] = rgb if isinstance(rgb, list) else []
    depth_list: list[Any] = depth if isinstance(depth, list) else []
    seen2: set[tuple[Any, Any]] = set()
    result2: list[Any] = []
    for item in rgb_list + depth_list:
        key = (item[0], type(item[1]).__name__)
        if key not in seen2:
            seen2.add(key)
            result2.append(item)
    return result2


def _convert_global_map(grid: Any) -> Any:
    return grid.to_rerun(voxel_size=0.1, mode="boxes")


def _convert_navigation_costmap(grid: Any) -> Any:
    return grid.to_rerun(
        colormap="Accent",
        z_offset=0.015,
        opacity=0.2,
        background="#484981",
    )


def _static_base_link(rr: Any) -> list[Any]:
    return [
        rr.Boxes3D(
            half_sizes=[0.35, 0.155, 0.2],
            colors=[(0, 255, 127)],
            fill_mode="MajorWireframe",
        ),
        rr.Transform3D(parent_frame="tf#/base_link"),
    ]


def _go2_rerun_blueprint() -> Any:
    """Split layout: camera feeds (Go2 + RealSense RGB/Depth) + 3D world view."""
    import rerun.blueprint as rrb

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/camera_optical/rgb", name="Go2 Camera"),
                rrb.Spatial2DView(origin="world/camera_optical/depth", name="Go2 Depth"),
            ),
            rrb.Vertical(
                rrb.Spatial2DView(origin="world/realsense_optical/rgb", name="RealSense RGB"),
                rrb.Spatial2DView(origin="world/realsense_optical/depth", name="RealSense Depth"),
            ),
            rrb.Spatial3DView(origin="world", name="3D"),
            column_shares=[1, 1, 2],
        ),
    )


rerun_config = {
    "blueprint": _go2_rerun_blueprint,
    # any pubsub that supports subscribe_all and topic that supports str(topic)
    # is acceptable here
    "pubsubs": [LCM()],
    # Custom converters for specific rerun entity paths
    # Normally all these would be specified in their respectative modules
    # Until this is implemented we have central overrides here
    #
    # This is unsustainable once we move to multi robot etc
    "visual_override": {
        "world/camera_info": _convert_camera_info,
        "world/realsense_camera_info": _convert_realsense_camera_info,
        "world/global_map": _convert_global_map,
        "world/navigation_costmap": _convert_navigation_costmap,
    },
    # slapping a go2 shaped box on top of tf/base_link
    "static": {
        "world/tf/base_link": _static_base_link,
    },
}


if global_config.viewer == "foxglove":
    from dimos.robot.foxglove_bridge import FoxgloveBridge

    with_vis = autoconnect(
        _transports_base,
        FoxgloveBridge.blueprint(shm_channels=["/color_image#sensor_msgs.Image"]),
    )
elif global_config.viewer.startswith("rerun"):
    from dimos.visualization.rerun.bridge import RerunBridgeModule, _resolve_viewer_mode

    with_vis = autoconnect(
        _transports_base,
        RerunBridgeModule.blueprint(viewer_mode=_resolve_viewer_mode(), **rerun_config),
    )
else:
    with_vis = _transports_base

unitree_go2_basic = (
    autoconnect(
        with_vis,
        GO2Connection.blueprint(),
        WebsocketVisModule.blueprint(),
    )
    .global_config(n_workers=4, robot_model="unitree_go2")
    .configurators(ClockSyncConfigurator())
)

__all__ = [
    "unitree_go2_basic",
]

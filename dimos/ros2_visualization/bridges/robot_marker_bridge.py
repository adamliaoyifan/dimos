# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""RobotMarkerBridge — interactive marker anchored at base_link.

Publishes a ``visualization_msgs/InteractiveMarker`` containing:
  - A box representing the robot's body (L × W × H).
  - A ``description`` JSON string with full geometry, mass, and sensor mounts.
    Foxglove and RViz2 display ``description`` in a tooltip when the marker
    is clicked / hovered.

The marker is published once at startup (latched).  Re-publish by calling
``on_sample()`` with a new ``RobotGeometry`` if the geometry changes.
"""

from __future__ import annotations

import json
import threading
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.schema import RobotGeometry


class RobotMarkerBridge(Bridge):
    """Converts ``RobotGeometry`` → latched ``InteractiveMarker``."""

    sample_type = RobotGeometry
    name = "robot_marker"

    def __init__(
        self,
        server_name: str = "robot_marker_server",
        frame: str = "base_link",
    ) -> None:
        super().__init__()
        self._server_name = server_name
        self._frame = frame
        self._server: Any = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from interactive_markers.interactive_marker_server import (  # type: ignore[import-untyped]
            InteractiveMarkerServer,
        )

        self._server = InteractiveMarkerServer(node, self._server_name)

    def stop(self) -> None:
        if self._server is not None:
            self._server.clear()
            self._server.applyChanges()
        super().stop()

    def on_sample(self, sample: RobotGeometry) -> None:
        self._check_started()

        from interactive_markers.interactive_marker_server import InteractiveMarkerServer  # type: ignore[import-untyped]
        from visualization_msgs.msg import (  # type: ignore[import-untyped]
            InteractiveMarker,
            InteractiveMarkerControl,
            Marker,
        )
        from dimos.ros2_visualization.core.clock import ClockPublisher, now_ns

        im = InteractiveMarker()
        im.header.frame_id = sample.base_link_frame or self._frame
        im.header.stamp = ClockPublisher.make_ros_time(now_ns())
        im.name = sample.name
        im.description = json.dumps(
            {
                "name": sample.name,
                "length_m": sample.length,
                "width_m": sample.width,
                "height_m": sample.height,
                "mass_kg": sample.mass_kg,
                "mesh": sample.mesh_resource,
                "sensors": sample.sensor_mounts,
            },
            default=str,
        )
        im.scale = max(sample.length, sample.width, sample.height) * 1.5

        body = Marker()
        body.type = Marker.CUBE
        body.scale.x = sample.length
        body.scale.y = sample.width
        body.scale.z = sample.height
        body.color.r = 0.3
        body.color.g = 0.6
        body.color.b = 0.9
        body.color.a = 0.5

        ctrl = InteractiveMarkerControl()
        ctrl.always_visible = True
        ctrl.markers.append(body)
        im.controls.append(ctrl)

        with self._lock:
            self._server.insert(im)
            self._server.applyChanges()

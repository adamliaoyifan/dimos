# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TrajectoryBridge — publishes NavDP (or any planner) trajectories.

Each ``TrajSample`` produces:
  - ``/viz/trajectory``          — ``visualization_msgs/MarkerArray`` (arrow per waypoint)
  - ``/viz/trajectory/metadata`` — ``std_msgs/String`` JSON keyed by marker id

The metadata topic enables the Foxglove custom panel to display per-waypoint
critic scores, costs, speeds, and world coordinates when a waypoint is clicked.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.metadata import MetadataSidecar
from dimos.ros2_visualization.core.schema import TrajSample


class TrajectoryBridge(Bridge):
    """Converts ``TrajSample`` → ``MarkerArray`` + metadata sidecar."""

    sample_type = TrajSample
    name = "trajectory"

    def __init__(self, topic: str = "/viz/trajectory") -> None:
        super().__init__()
        self._topic = topic
        self._pub: Any = None
        self._meta: MetadataSidecar | None = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from visualization_msgs.msg import Marker, MarkerArray  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub = node.create_publisher(
            MarkerArray, self._topic, build(QoSProfile.RELIABLE)
        )
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self._meta = MetadataSidecar(node, self._topic)

    def on_sample(self, sample: TrajSample) -> None:
        self._check_started()

        from dimos.ros2_visualization.core.clock import ClockPublisher

        stamp = ClockPublisher.make_ros_time(sample.stamp_ns)
        markers: list[Any] = []
        metadata: dict[int, dict] = {}

        r, g, b = sample.color_rgb
        alpha = 0.9 if sample.is_selected else 0.4

        for idx, (pt, meta) in enumerate(zip(sample.points, sample.waypoint_meta)):
            m = self._Marker()
            m.header.stamp = stamp
            m.header.frame_id = sample.frame
            m.ns = f"traj_{sample.traj_id}"
            m.id = idx
            m.type = self._Marker.ARROW
            m.action = self._Marker.ADD

            m.pose.position.x = pt.x
            m.pose.position.y = pt.y
            m.pose.orientation.z = math.sin(pt.yaw / 2.0)
            m.pose.orientation.w = math.cos(pt.yaw / 2.0)

            m.scale.x = 0.12  # arrow shaft length
            m.scale.y = 0.04  # arrow width
            m.scale.z = 0.04

            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = alpha

            markers.append(m)

            metadata[idx] = {
                "traj_id": sample.traj_id,
                "waypoint": idx,
                "x": round(pt.x, 4),
                "y": round(pt.y, 4),
                "yaw_deg": round(math.degrees(pt.yaw), 2),
                "critic": round(meta.critic_score, 4),
                "cost": round(meta.cost, 4),
                "speed": round(meta.speed, 4),
                "selected": sample.is_selected,
                "source": sample.source,
                **{k: v for k, v in meta.extra.items()},
            }

        arr = self._MarkerArray()
        arr.markers = markers

        with self._lock:
            self._pub.publish(arr)
            if self._meta is not None:
                self._meta.publish(metadata)

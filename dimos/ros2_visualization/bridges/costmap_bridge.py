# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CostmapBridge — publishes nav_msgs/OccupancyGrid on ``/viz/costmap``
and overlays frontier candidates as ``visualization_msgs/MarkerArray``
on ``/viz/frontiers``.
"""

from __future__ import annotations

import threading
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.metadata import MetadataSidecar
from dimos.ros2_visualization.core.schema import CostmapSample, FrontierSample


class CostmapBridge(Bridge):
    """Converts ``CostmapSample`` → ``nav_msgs/OccupancyGrid``."""

    sample_type = CostmapSample
    name = "costmap"

    def __init__(self, topic: str = "/viz/costmap") -> None:
        super().__init__()
        self._topic = topic
        self._pub: Any = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from nav_msgs.msg import MapMetaData, OccupancyGrid  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub = node.create_publisher(
            OccupancyGrid, self._topic, build(QoSProfile.RELIABLE)
        )
        self._OccupancyGrid = OccupancyGrid
        self._MapMetaData = MapMetaData

    def on_sample(self, sample: CostmapSample) -> None:
        self._check_started()

        from dimos.ros2_visualization.core.clock import ClockPublisher

        msg = self._OccupancyGrid()
        stamp = ClockPublisher.make_ros_time(sample.stamp_ns)

        msg.header.stamp = stamp
        msg.header.frame_id = sample.frame

        msg.info.resolution = sample.resolution
        msg.info.width = sample.width
        msg.info.height = sample.height
        msg.info.origin.position.x = sample.origin_x
        msg.info.origin.position.y = sample.origin_y
        msg.info.origin.orientation.w = 1.0

        msg.data = list(sample.data)

        with self._lock:
            self._pub.publish(msg)


class FrontierBridge(Bridge):
    """Converts ``FrontierSample`` → ``visualization_msgs/MarkerArray`` on ``/viz/frontiers``."""

    sample_type = FrontierSample
    name = "frontiers"

    def __init__(self, topic: str = "/viz/frontiers") -> None:
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

    def on_sample(self, sample: FrontierSample) -> None:
        self._check_started()

        from dimos.ros2_visualization.core.clock import ClockPublisher

        stamp = ClockPublisher.make_ros_time(sample.stamp_ns)
        markers = []
        metadata: dict[int, dict] = {}

        for idx, f in enumerate(sample.frontiers):
            m = self._Marker()
            m.header.stamp = stamp
            m.header.frame_id = sample.frame
            m.ns = "frontiers"
            m.id = idx
            m.type = self._Marker.SPHERE
            m.action = self._Marker.ADD
            m.pose.position.x = f.x
            m.pose.position.y = f.y
            m.pose.orientation.w = 1.0

            # Size scales with score
            s = max(0.1, min(0.5, f.score * 0.5))
            m.scale.x = s
            m.scale.y = s
            m.scale.z = 0.05

            # Colour: green (high score) → red (low score)
            m.color.a = 0.85
            m.color.r = 1.0 - f.score
            m.color.g = f.score
            m.color.b = 0.1

            markers.append(m)

            metadata[idx] = {
                "x": round(f.x, 3),
                "y": round(f.y, 3),
                "score": round(f.score, 4),
                "info_gain": round(f.info_gain, 4),
                "memory_novelty": round(f.memory_novelty, 4),
                "corridor_score": round(f.corridor_score, 4),
                "vlm_confidence": f.vlm_confidence,
                "rank": f.rank,
            }

        arr = self._MarkerArray()
        arr.markers = markers

        with self._lock:
            self._pub.publish(arr)
            if self._meta is not None:
                self._meta.publish(metadata)

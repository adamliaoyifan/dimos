# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PathBridge — publishes nav_msgs/Path on ``/viz/path``.

Maintains a grow-only history trail (up to ``max_poses`` poses).  Each new
``PathSample`` appends its poses; old poses beyond the cap are dropped from
the front.
"""

from __future__ import annotations

import math
import threading
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.schema import OdomSample, PathSample


class PathBridge(Bridge):
    """Accumulates robot pose history and publishes as ``nav_msgs/Path``."""

    sample_type = PathSample
    name = "path"

    def __init__(self, topic: str = "/viz/path", max_poses: int = 2000) -> None:
        super().__init__()
        self._topic = topic
        self._max_poses = max_poses
        self._history: list[Any] = []
        self._pub: Any = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from geometry_msgs.msg import PoseStamped  # type: ignore[import-untyped]
        from nav_msgs.msg import Path  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub = node.create_publisher(Path, self._topic, build(QoSProfile.RELIABLE))
        self._Path = Path
        self._PoseStamped = PoseStamped

    def on_sample(self, sample: PathSample) -> None:
        self._check_started()

        from dimos.ros2_visualization.core.clock import ClockPublisher

        with self._lock:
            for p in sample.poses:
                ps = self._PoseStamped()
                ps.header.stamp = ClockPublisher.make_ros_time(p.stamp_ns)
                ps.header.frame_id = p.frame
                ps.pose.position.x = p.x
                ps.pose.position.y = p.y
                ps.pose.orientation.z = math.sin(p.yaw / 2.0)
                ps.pose.orientation.w = math.cos(p.yaw / 2.0)
                self._history.append(ps)

            if len(self._history) > self._max_poses:
                self._history = self._history[-self._max_poses :]

            msg = self._Path()
            if self._history:
                msg.header = self._history[-1].header
                msg.poses = list(self._history)
            self._pub.publish(msg)


class OdomPathBridge(Bridge):
    """Convenience bridge that builds the path directly from ``OdomSample`` stream.

    Register this **instead of** ``PathBridge`` if your adapter publishes
    ``OdomSample`` but not ``PathSample``.
    """

    sample_type = OdomSample
    name = "odom_path"

    def __init__(self, topic: str = "/viz/path", max_poses: int = 2000) -> None:
        super().__init__()
        self._topic = topic
        self._max_poses = max_poses
        self._history: list[Any] = []
        self._pub: Any = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from geometry_msgs.msg import PoseStamped  # type: ignore[import-untyped]
        from nav_msgs.msg import Path  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub = node.create_publisher(Path, self._topic, build(QoSProfile.RELIABLE))
        self._Path = Path
        self._PoseStamped = PoseStamped

    def on_sample(self, sample: OdomSample) -> None:
        self._check_started()

        from dimos.ros2_visualization.core.clock import ClockPublisher

        ps = self._PoseStamped()
        ps.header.stamp = ClockPublisher.make_ros_time(sample.pose.stamp_ns)
        ps.header.frame_id = sample.pose.frame
        ps.pose.position.x = sample.pose.x
        ps.pose.position.y = sample.pose.y
        ps.pose.position.z = sample.pose.z
        ps.pose.orientation.x = sample.pose.qx
        ps.pose.orientation.y = sample.pose.qy
        ps.pose.orientation.z = sample.pose.qz
        ps.pose.orientation.w = sample.pose.qw

        with self._lock:
            self._history.append(ps)
            if len(self._history) > self._max_poses:
                self._history = self._history[-self._max_poses :]

            msg = self._Path()
            msg.header = ps.header
            msg.poses = list(self._history)
            self._pub.publish(msg)

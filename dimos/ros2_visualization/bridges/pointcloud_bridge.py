# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""PointCloudBridge — publishes sensor_msgs/PointCloud2 on ``/viz/pointcloud``.

Rate-limited to ``max_fps`` (default 5) — LiDAR scans at 10 Hz are heavy.
"""

from __future__ import annotations

import struct
import threading
import time
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.schema import PointCloudSample


class PointCloudBridge(Bridge):
    """Converts ``PointCloudSample`` → ``sensor_msgs/PointCloud2``."""

    sample_type = PointCloudSample
    name = "pointcloud"

    def __init__(self, topic: str = "/viz/pointcloud", max_fps: float = 5.0) -> None:
        super().__init__()
        self._topic = topic
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._last_publish: float = 0.0
        self._pub: Any = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from sensor_msgs.msg import PointCloud2, PointField  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub = node.create_publisher(
            PointCloud2, self._topic, build(QoSProfile.SENSOR_DATA)
        )
        self._PointCloud2 = PointCloud2
        self._PointField = PointField

    def on_sample(self, sample: PointCloudSample) -> None:
        self._check_started()

        now = time.monotonic()
        with self._lock:
            if now - self._last_publish < self._min_interval:
                return
            self._last_publish = now

        from dimos.ros2_visualization.core.clock import ClockPublisher

        msg = self._PointCloud2()
        msg.header.stamp = ClockPublisher.make_ros_time(sample.stamp_ns)
        msg.header.frame_id = sample.frame

        float_size = 4
        point_step = 3 * float_size  # x, y, z

        has_intensity = sample.intensity is not None
        if has_intensity:
            point_step += float_size

        fields = [
            self._PointField(name="x", offset=0, datatype=7, count=1),   # FLOAT32
            self._PointField(name="y", offset=4, datatype=7, count=1),
            self._PointField(name="z", offset=8, datatype=7, count=1),
        ]
        if has_intensity:
            fields.append(
                self._PointField(name="intensity", offset=12, datatype=7, count=1)
            )

        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = point_step
        msg.height = 1
        msg.width = sample.num_points

        if has_intensity:
            # Interleave xyz + intensity
            xyz = sample.points_xyz
            intensity = sample.intensity
            data = bytearray(sample.num_points * point_step)
            for i in range(sample.num_points):
                off_out = i * point_step
                off_xyz = i * 12
                data[off_out : off_out + 12] = xyz[off_xyz : off_xyz + 12]
                off_i = i * 4
                data[off_out + 12 : off_out + 16] = intensity[off_i : off_i + 4]  # type: ignore[index]
            msg.data = bytes(data)
        else:
            msg.data = sample.points_xyz

        msg.row_step = msg.width * msg.point_step
        msg.is_dense = True

        with self._lock:
            self._pub.publish(msg)

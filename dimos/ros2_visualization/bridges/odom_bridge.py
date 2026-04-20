# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""OdomBridge — publishes nav_msgs/Odometry on ``/viz/odom``.

Also updates the shared :class:`~dimos.ros2_visualization.core.tf_broker.TFBroker`
so the TF tree is always in sync with the latest odometry reading.
"""

from __future__ import annotations

import threading
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.schema import OdomSample


class OdomBridge(Bridge):
    """Converts ``OdomSample`` → ``nav_msgs/Odometry`` and ``/tf``."""

    sample_type = OdomSample
    name = "odom"

    def __init__(self, topic: str = "/viz/odom", tf_broker: Any = None) -> None:
        super().__init__()
        self._topic = topic
        self._tf_broker = tf_broker
        self._pub: Any = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from nav_msgs.msg import Odometry  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub = node.create_publisher(Odometry, self._topic, build(QoSProfile.RELIABLE))
        self._Odometry = Odometry

    def on_sample(self, sample: OdomSample) -> None:
        self._check_started()

        from dimos.ros2_visualization.core.clock import ClockPublisher

        msg = self._Odometry()
        stamp = ClockPublisher.make_ros_time(sample.pose.stamp_ns)

        msg.header.stamp = stamp
        msg.header.frame_id = sample.pose.frame
        msg.child_frame_id = sample.child_frame

        msg.pose.pose.position.x = sample.pose.x
        msg.pose.pose.position.y = sample.pose.y
        msg.pose.pose.position.z = sample.pose.z
        msg.pose.pose.orientation.x = sample.pose.qx
        msg.pose.pose.orientation.y = sample.pose.qy
        msg.pose.pose.orientation.z = sample.pose.qz
        msg.pose.pose.orientation.w = sample.pose.qw

        msg.twist.twist.linear.x = sample.vx
        msg.twist.twist.linear.y = sample.vy
        msg.twist.twist.linear.z = sample.vz
        msg.twist.twist.angular.x = sample.wx
        msg.twist.twist.angular.y = sample.wy
        msg.twist.twist.angular.z = sample.wz

        with self._lock:
            self._pub.publish(msg)

        if self._tf_broker is not None:
            self._tf_broker.update(sample)

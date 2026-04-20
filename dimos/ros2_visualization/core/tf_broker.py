# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TF tree manager.

``TFBroker`` publishes:
  - Dynamic: ``map → odom → base_link`` derived from each ``OdomSample``
  - Static:  ``base_link → lidar``,  ``base_link → camera``  (once at startup)

All frames are driven from the ``FrameIds`` passed into the constructor so the
same broker works for any robot.
"""

from __future__ import annotations

import math
from typing import Any

from dimos.ros2_visualization.core.frames import FrameIds
from dimos.ros2_visualization.core.schema import OdomSample


class TFBroker:
    """Publishes ``/tf`` and ``/tf_static`` from OdomSamples.

    Call ``start(node)`` once, then ``update(odom)`` on every new odometry
    reading.
    """

    def __init__(self, frames: FrameIds | None = None) -> None:
        self._frames = frames or FrameIds()
        self._node: Any = None
        self._dyn_pub: Any = None
        self._static_pub: Any = None

    def start(self, node: Any) -> None:
        from geometry_msgs.msg import TransformStamped  # type: ignore[import-untyped]
        from tf2_ros import TransformBroadcaster, StaticTransformBroadcaster  # type: ignore[import-untyped]

        self._node = node
        self._tf_broadcaster = TransformBroadcaster(node)
        self._static_broadcaster = StaticTransformBroadcaster(node)
        self._TransformStamped = TransformStamped

        self._publish_static_transforms()

    def _publish_static_transforms(self) -> None:
        from dimos.ros2_visualization.core.clock import ClockPublisher, now_ns

        now = ClockPublisher.make_ros_time(now_ns())
        transforms = []

        lidar_tf = self._TransformStamped()
        lidar_tf.header.stamp = now
        lidar_tf.header.frame_id = self._frames.base_link
        lidar_tf.child_frame_id = self._frames.lidar
        ox, oy, oz = self._frames.lidar_offset_xyz
        lidar_tf.transform.translation.x = ox
        lidar_tf.transform.translation.y = oy
        lidar_tf.transform.translation.z = oz
        lidar_tf.transform.rotation.w = 1.0
        transforms.append(lidar_tf)

        cam_tf = self._TransformStamped()
        cam_tf.header.stamp = now
        cam_tf.header.frame_id = self._frames.base_link
        cam_tf.child_frame_id = self._frames.camera
        cx, cy, cz = self._frames.camera_offset_xyz
        cam_tf.transform.translation.x = cx
        cam_tf.transform.translation.y = cy
        cam_tf.transform.translation.z = cz
        cam_tf.transform.rotation.w = 1.0
        transforms.append(cam_tf)

        self._static_broadcaster.sendTransform(transforms)

    def update(self, odom: OdomSample) -> None:
        """Publish dynamic TF chain from *odom*."""
        if self._node is None:
            return

        from dimos.ros2_visualization.core.clock import ClockPublisher

        stamp = ClockPublisher.make_ros_time(odom.pose.stamp_ns)
        transforms = []

        # map → odom (identity — we treat odom as map for now)
        map_odom = self._TransformStamped()
        map_odom.header.stamp = stamp
        map_odom.header.frame_id = self._frames.map
        map_odom.child_frame_id = self._frames.odom
        map_odom.transform.rotation.w = 1.0
        transforms.append(map_odom)

        # odom → base_link
        odom_base = self._TransformStamped()
        odom_base.header.stamp = stamp
        odom_base.header.frame_id = self._frames.odom
        odom_base.child_frame_id = self._frames.base_link
        odom_base.transform.translation.x = odom.pose.x
        odom_base.transform.translation.y = odom.pose.y
        odom_base.transform.translation.z = odom.pose.z
        odom_base.transform.rotation.x = odom.pose.qx
        odom_base.transform.rotation.y = odom.pose.qy
        odom_base.transform.rotation.z = odom.pose.qz
        odom_base.transform.rotation.w = odom.pose.qw
        transforms.append(odom_base)

        self._tf_broadcaster.sendTransform(transforms)

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ImageBridge — publishes sensor_msgs/Image or CompressedImage on ``/viz/image``.

Heavy images are rate-limited to ``max_fps`` (default 10) to avoid overwhelming
the Foxglove WebSocket bridge.  JPEG-encoded ``ImageSample`` values are forwarded
directly as ``sensor_msgs/CompressedImage``; raw RGB/BGR samples are published as
``sensor_msgs/Image``.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.schema import ImageSample


class ImageBridge(Bridge):
    """Converts ``ImageSample`` → ``sensor_msgs/Image`` or ``CompressedImage``."""

    sample_type = ImageSample
    name = "image"

    def __init__(
        self,
        topic: str = "/viz/image",
        max_fps: float = 10.0,
        compress: bool = True,
    ) -> None:
        super().__init__()
        self._topic = topic
        self._min_interval = 1.0 / max_fps if max_fps > 0 else 0.0
        self._compress = compress
        self._last_publish: float = 0.0
        self._pub: Any = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from dimos.ros2_visualization.core.qos import QoSProfile, build

        if self._compress:
            from sensor_msgs.msg import CompressedImage  # type: ignore[import-untyped]

            self._pub = node.create_publisher(
                CompressedImage,
                f"{self._topic}/compressed",
                build(QoSProfile.SENSOR_DATA),
            )
            self._MsgClass = CompressedImage
        else:
            from sensor_msgs.msg import Image  # type: ignore[import-untyped]

            self._pub = node.create_publisher(
                Image,
                self._topic,
                build(QoSProfile.SENSOR_DATA),
            )
            self._MsgClass = Image

    def on_sample(self, sample: ImageSample) -> None:
        self._check_started()

        now = time.monotonic()
        with self._lock:
            if now - self._last_publish < self._min_interval:
                return
            self._last_publish = now

        from dimos.ros2_visualization.core.clock import ClockPublisher

        stamp = ClockPublisher.make_ros_time(sample.stamp_ns)

        if self._compress:
            msg = self._MsgClass()
            msg.header.stamp = stamp
            msg.header.frame_id = sample.frame
            if sample.encoding == "jpeg":
                msg.format = "jpeg"
                msg.data = list(sample.data)
            else:
                import cv2  # type: ignore[import-untyped]
                import numpy as np

                arr = np.frombuffer(sample.data, dtype=np.uint8).reshape(
                    sample.height, sample.width, -1
                )
                if sample.encoding == "rgb8":
                    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
                _, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 80])
                msg.format = "jpeg"
                msg.data = buf.tobytes()
        else:
            msg = self._MsgClass()
            msg.header.stamp = stamp
            msg.header.frame_id = sample.frame
            msg.height = sample.height
            msg.width = sample.width
            msg.encoding = sample.encoding
            msg.step = sample.width * (3 if "8" in sample.encoding else 1)
            msg.data = list(sample.data)

        with self._lock:
            self._pub.publish(msg)

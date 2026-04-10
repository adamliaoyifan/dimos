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

"""Relay LCM ``sensor_msgs/CompressedImage`` → realsense pSHM as DimOS ``Image`` (no rclpy).

Use with ``navdp.trajectory_camera: realsense_lcm`` and ``ros2_dimos_lcm_bridge``,
which publishes JPEG colour and PNG depth as LCM ``CompressedImage`` on the
configured basenames (``basename#sensor_msgs.CompressedImage``).
"""

from __future__ import annotations

import logging
from typing import Any

from dimos_lcm.sensor_msgs.CompressedImage import CompressedImage
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import In, Out
from dimos.hardware.sensors.camera.compressed_image_decode import (
    decode_jpeg_to_rgb_image,
    decode_png_depth_to_image,
    normalize_compressed_payload,
)
from dimos.msgs.sensor_msgs.Image import Image

logger = logging.getLogger(__name__)


class LcmRealsenseRelayConfig(ModuleConfig):
    """LCM basenames; full channel is ``basename#sensor_msgs.CompressedImage``."""

    lcm_rgb_basename: str = "/dimos/realsense/rgb"
    lcm_depth_basename: str = "/dimos/realsense/depth"
    # Divisor to convert raw uint16 depth to float32 metres (same as ROS2 bridge).
    depth_scale: float = 1000.0


class LcmRealsenseRelay(Module[LcmRealsenseRelayConfig]):
    """Subscribe LCM CompressedImage (RGB JPEG + depth PNG), publish ``realsense_*`` pSHM."""

    rgb_ingress: In[CompressedImage]
    depth_ingress: In[CompressedImage]
    realsense_image: Out[Image]
    realsense_depth: Out[Image]

    default_config = LcmRealsenseRelayConfig

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(
            Disposable(self.rgb_ingress.subscribe(self._on_rgb)),
        )
        self._disposables.add(
            Disposable(self.depth_ingress.subscribe(self._on_depth)),
        )
        logger.info(
            "[LcmRealsenseRelay] LCM rgb=%s depth=%s → pSHM realsense_image / realsense_depth",
            self.config.lcm_rgb_basename,
            self.config.lcm_depth_basename,
        )

    def _on_rgb(self, msg: CompressedImage) -> None:
        payload = normalize_compressed_payload(msg.data)
        img = decode_jpeg_to_rgb_image(payload, log_prefix="[LcmRealsenseRelay]")
        if img is not None:
            self.realsense_image.publish(img)

    def _on_depth(self, msg: CompressedImage) -> None:
        payload = normalize_compressed_payload(msg.data)
        img = decode_png_depth_to_image(
            payload,
            self.config.depth_scale,
            log_prefix="[LcmRealsenseRelay]",
        )
        if img is not None:
            self.realsense_depth.publish(img)

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)


__all__ = ["LcmRealsenseRelay", "LcmRealsenseRelayConfig"]

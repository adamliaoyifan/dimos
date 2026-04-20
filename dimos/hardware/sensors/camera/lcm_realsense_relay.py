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

Also relays LCM ``sensor_msgs/CameraInfo`` → ``realsense_camera_info`` out-stream, and
exposes a threading ``Event`` (``camera_info_ready``) that other modules can block on
until the first CameraInfo arrives from the bridge.

Use with ``navdp.trajectory_camera: realsense_lcm`` and ``ros2_dimos_lcm_bridge``,
which publishes JPEG colour and PNG depth as LCM ``CompressedImage`` on the
configured basenames (``basename#sensor_msgs.CompressedImage``), and CameraInfo on
``lcm_camera_info_basename#sensor_msgs.CameraInfo``.
"""

from __future__ import annotations

import logging
import threading
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
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image

logger = logging.getLogger(__name__)


class LcmRealsenseRelayConfig(ModuleConfig):
    """LCM basenames; full channel is ``basename#sensor_msgs.CompressedImage``."""

    lcm_rgb_basename: str = "/dimos/realsense/rgb"
    lcm_depth_basename: str = "/dimos/realsense/depth"
    lcm_camera_info_basename: str = "/dimos/realsense/camera_info"
    # Divisor to convert raw uint16 depth to float32 metres (same as ROS2 bridge).
    depth_scale: float = 1000.0


class LcmRealsenseRelay(Module[LcmRealsenseRelayConfig]):
    """Subscribe LCM CompressedImage (RGB JPEG + depth PNG) and CameraInfo, publish pSHM streams.

    Streams
    -------
    rgb_ingress : In[CompressedImage]
        LCM JPEG colour frames from the bridge.
    depth_ingress : In[CompressedImage]
        LCM PNG depth frames from the bridge.
    camera_info_ingress : In[CameraInfo]
        LCM CameraInfo from the bridge (forwarded as-is).
    realsense_image : Out[Image]
        Decoded RGB image on pSHM.
    realsense_depth : Out[Image]
        Decoded depth image (float32 metres) on pSHM.
    realsense_camera_info : Out[CameraInfo]
        Forwarded CameraInfo for downstream consumers (Navigator, GO2Connection, VLN).

    Attributes
    ----------
    camera_info_ready : threading.Event
        Set when the first CameraInfo message arrives. Other modules can call
        ``camera_info_ready.wait(timeout=...)`` to block until intrinsics are known.
    live_camera_info : CameraInfo | None
        The most recently received CameraInfo, or None before the first message.
    """

    rgb_ingress: In[CompressedImage]
    depth_ingress: In[CompressedImage]
    camera_info_ingress: In[CameraInfo]
    realsense_image: Out[Image]
    realsense_depth: Out[Image]
    realsense_camera_info: Out[CameraInfo]

    default_config = LcmRealsenseRelayConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.camera_info_ready: threading.Event = threading.Event()
        self.live_camera_info: CameraInfo | None = None

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(Disposable(self.rgb_ingress.subscribe(self._on_rgb)))
        self._disposables.add(Disposable(self.depth_ingress.subscribe(self._on_depth)))
        self._disposables.add(Disposable(self.camera_info_ingress.subscribe(self._on_camera_info)))
        logger.info(
            "[LcmRealsenseRelay] LCM rgb=%s depth=%s camera_info=%s → pSHM",
            self.config.lcm_rgb_basename,
            self.config.lcm_depth_basename,
            self.config.lcm_camera_info_basename,
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

    def _on_camera_info(self, msg: CameraInfo) -> None:
        self.live_camera_info = msg
        self.realsense_camera_info.publish(msg)
        if not self.camera_info_ready.is_set():
            logger.info(
                "[LcmRealsenseRelay] CameraInfo received: %dx%d fx=%.1f fy=%.1f",
                msg.width,
                msg.height,
                msg.K[0] if msg.K else 0.0,
                msg.K[4] if msg.K else 0.0,
            )
            self.camera_info_ready.set()


__all__ = ["LcmRealsenseRelay", "LcmRealsenseRelayConfig"]

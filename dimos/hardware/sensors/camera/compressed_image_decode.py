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

"""Decode ROS / LCM ``sensor_msgs/CompressedImage`` payloads to DimOS ``Image``.

Shared by ``ROS2CompressedImageBridge`` and ``LcmRealsenseRelay``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np

from dimos.msgs.sensor_msgs.Image import Image, ImageFormat

logger = logging.getLogger(__name__)


def normalize_compressed_payload(data: Any) -> bytes:
    """Coerce ``CompressedImage.data`` (bytes, memoryview, or buffer) to ``bytes``."""
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, bytes):
        return data
    return bytes(data)


def decode_jpeg_to_rgb_image(
    data: bytes, *, log_prefix: str = "[compressed_decode]"
) -> Image | None:
    """Decompress JPEG bytes → RGB ``Image`` (``ImageFormat.RGB``)."""
    try:
        np_arr = np.frombuffer(data, np.uint8)
        bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if bgr is None:
            logger.warning("%s RGB imdecode returned None", log_prefix)
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return Image(data=rgb, format=ImageFormat.RGB, ts=time.time())
    except Exception as e:
        logger.warning("%s RGB decode error: %s", log_prefix, e)
        return None


def decode_png_depth_to_image(
    data: bytes, depth_scale: float, *, log_prefix: str = "[compressed_decode]"
) -> Image | None:
    """Decompress PNG (typically uint16 depth) → float32 metres (``ImageFormat.DEPTH``)."""
    try:
        np_arr = np.frombuffer(data, np.uint8)
        depth_raw = cv2.imdecode(np_arr, cv2.IMREAD_UNCHANGED)
        if depth_raw is None:
            logger.warning("%s depth imdecode returned None", log_prefix)
            return None
        if depth_raw.ndim == 3:
            depth_raw = cv2.cvtColor(depth_raw, cv2.COLOR_BGR2GRAY)
        depth_m = depth_raw.astype(np.float32) / depth_scale
        return Image(data=depth_m, format=ImageFormat.DEPTH, ts=time.time())
    except Exception as e:
        logger.warning("%s depth decode error: %s", log_prefix, e)
        return None


__all__ = [
    "decode_jpeg_to_rgb_image",
    "decode_png_depth_to_image",
    "normalize_compressed_payload",
]

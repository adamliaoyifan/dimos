"""Encode ROS 2 image messages into LCM CompressedImage bytes."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from dimos_lcm.sensor_msgs.CameraInfo import CameraInfo as LCMCameraInfo
from dimos_lcm.sensor_msgs.CompressedImage import CompressedImage as LCMCompressedImage
from dimos_lcm.std_msgs.Header import Header
from dimos_lcm.std_msgs.Time import Time


def _make_header(ros_header: Any) -> Header:
    h = Header()
    h.seq = getattr(ros_header, "seq", 0)
    h.frame_id = getattr(ros_header, "frame_id", "")
    stamp = getattr(ros_header, "stamp", None)
    if stamp is not None:
        t = Time()
        t.sec = stamp.sec
        t.nsec = stamp.nanosec
        h.stamp = t
    return h


def ros_image_rgb_to_lcm_bytes(msg: Any, jpeg_quality: int = 85) -> bytes:
    """Convert a ROS 2 ``sensor_msgs/Image`` (BGR8/RGB8) to LCM CompressedImage (JPEG)."""
    encoding = msg.encoding.lower()
    raw = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)
    # crop to actual width
    if encoding in ("rgb8",):
        frame = raw[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    else:
        # bgr8 or unknown — pass as-is
        frame = raw[:, : msg.width * 3].reshape(msg.height, msg.width, 3)

    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    payload = buf.tobytes()

    lcm_msg = LCMCompressedImage()
    lcm_msg.header = _make_header(msg.header)
    lcm_msg.format = "jpeg"
    lcm_msg.data = payload
    lcm_msg.data_length = len(payload)
    return lcm_msg.lcm_encode()


def ros_image_depth_to_lcm_bytes(msg: Any) -> bytes:
    """Convert a ROS 2 ``sensor_msgs/Image`` (16UC1 depth) to LCM CompressedImage (PNG)."""
    raw = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.width)
    ok, buf = cv2.imencode(".png", raw)
    if not ok:
        raise RuntimeError("PNG encode failed")
    payload = buf.tobytes()

    lcm_msg = LCMCompressedImage()
    lcm_msg.header = _make_header(msg.header)
    lcm_msg.format = "png"
    lcm_msg.data = payload
    lcm_msg.data_length = len(payload)
    return lcm_msg.lcm_encode()


def ros_camera_info_to_lcm_bytes(msg: Any) -> bytes:
    """Copy ROS 2 ``sensor_msgs/CameraInfo`` fields into LCM ``CameraInfo`` and encode."""
    lcm_msg = LCMCameraInfo()
    lcm_msg.header = _make_header(msg.header)
    lcm_msg.height = msg.height
    lcm_msg.width = msg.width
    lcm_msg.distortion_model = msg.distortion_model
    lcm_msg.D = list(msg.d)
    lcm_msg.D_length = len(lcm_msg.D)
    lcm_msg.K = list(msg.k)
    lcm_msg.R = list(msg.r)
    lcm_msg.P = list(msg.p)
    lcm_msg.binning_x = msg.binning_x
    lcm_msg.binning_y = msg.binning_y
    return lcm_msg.lcm_encode()


def ros_compressed_image_to_lcm_bytes(msg: Any) -> bytes:
    """Copy ROS 2 ``CompressedImage`` fields into ``dimos_lcm`` and ``lcm_encode``.

    No decode/recompress — ``data`` and ``format`` are forwarded unchanged.
    """
    lcm_msg = LCMCompressedImage()
    lcm_msg.header = _make_header(msg.header)
    lcm_msg.format = msg.format
    lcm_msg.data = bytes(msg.data)
    lcm_msg.data_length = len(lcm_msg.data)
    return lcm_msg.lcm_encode()

"""CLI: source ROS 2, then run ``ros2-dimos-lcm-bridge`` (or ``python -m ros2_dimos_lcm_bridge``)."""

from __future__ import annotations

import argparse
import logging
import os
import sys

from ros2_dimos_lcm_bridge.encode import (
    ros_camera_info_to_lcm_bytes,
    ros_compressed_image_to_lcm_bytes,
    ros_image_depth_to_lcm_bytes,
    ros_image_rgb_to_lcm_bytes,
)


logger = logging.getLogger(__name__)

_DEFAULT_LCM_URL = os.environ.get("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=0")

_COMPRESSED_SUFFIX = "#sensor_msgs.CompressedImage"
_CAMERA_INFO_SUFFIX = "#sensor_msgs.CameraInfo"


def _lcm_channel(basename: str, suffix: str) -> str:
    b = basename.rstrip("/")
    if "#" in b:
        return b
    return f"{b}{suffix}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bridge ROS 2 raw/compressed image topics → LCM CompressedImage."
    )
    parser.add_argument("--lcm-url", default=_DEFAULT_LCM_URL)

    # ── RGB ──────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ros-rgb-topic",
        default="/camera/camera/color/image_raw",
        help="ROS 2 RGB topic (sensor_msgs/Image or CompressedImage)",
    )
    parser.add_argument(
        "--ros-rgb-compressed",
        action="store_true",
        help="RGB topic is already CompressedImage (skip JPEG re-encode)",
    )
    parser.add_argument(
        "--lcm-rgb-basename",
        default="/dimos/realsense/rgb",
        help="LCM channel basename for RGB frames",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=85,
        help="JPEG quality when encoding raw RGB (default 85)",
    )

    # ── Depth ─────────────────────────────────────────────────────────────────
    parser.add_argument(
        "--ros-depth-topic",
        default="/camera/camera/aligned_depth_to_color/image_raw",
        help="ROS 2 depth topic (sensor_msgs/Image 16UC1 or CompressedImage)",
    )
    parser.add_argument(
        "--ros-depth-compressed",
        action="store_true",
        help="Depth topic is already CompressedImage (skip PNG re-encode)",
    )
    parser.add_argument(
        "--lcm-depth-basename",
        default="/dimos/realsense/depth",
        help="LCM channel basename for depth frames",
    )
    parser.add_argument(
        "--depth-scale",
        type=float,
        default=1000.0,
        help="uint16 → metres divisor stored in LCM header (informational, default 1000)",
    )

    # ── Camera info ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--ros-camera-info-topic",
        default="/camera/camera/color/camera_info",
        help="ROS 2 CameraInfo topic (default: /camera/camera/color/camera_info)",
    )
    parser.add_argument(
        "--lcm-camera-info-basename",
        default="/dimos/realsense/camera_info",
        help="LCM channel basename for CameraInfo",
    )
    parser.add_argument(
        "--no-camera-info",
        action="store_true",
        help="Disable the camera_info bridge",
    )

    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # ── imports that require ROS 2 to be sourced ───────────────────────────
    try:
        import lcm
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import CameraInfo as RosCameraInfo
        from sensor_msgs.msg import CompressedImage as RosCompressedImage
        from sensor_msgs.msg import Image as RosImage
    except ImportError as exc:
        sys.exit(
            f"Import error — did you source ROS 2?\n  {exc}"
        )

    lc = lcm.LCM(args.lcm_url)
    logger.info("LCM URL: %s", args.lcm_url)

    rgb_channel = _lcm_channel(args.lcm_rgb_basename, _COMPRESSED_SUFFIX)
    depth_channel = _lcm_channel(args.lcm_depth_basename, _COMPRESSED_SUFFIX)
    camera_info_channel = _lcm_channel(args.lcm_camera_info_basename, _CAMERA_INFO_SUFFIX)

    rclpy.init()
    node = Node("ros2_dimos_lcm_bridge")

    best_effort_qos = QoSProfile(
        depth=1,
        reliability=QoSReliabilityPolicy.BEST_EFFORT,
    )

    # ── RGB subscriber ────────────────────────────────────────────────────────
    def _on_rgb_raw(msg: "RosImage") -> None:
        try:
            data = ros_image_rgb_to_lcm_bytes(msg, jpeg_quality=args.jpeg_quality)
            lc.publish(rgb_channel, data)
            logger.debug("RGB raw → LCM %s (%d B)", rgb_channel, len(data))
        except Exception:
            logger.exception("RGB encode error")

    def _on_rgb_compressed(msg: "RosCompressedImage") -> None:
        try:
            data = ros_compressed_image_to_lcm_bytes(msg)
            lc.publish(rgb_channel, data)
            logger.debug("RGB compressed → LCM %s (%d B)", rgb_channel, len(data))
        except Exception:
            logger.exception("RGB forward error")

    if args.ros_rgb_compressed:
        node.create_subscription(RosCompressedImage, args.ros_rgb_topic, _on_rgb_compressed, best_effort_qos)
        logger.info("ROS2 %s (CompressedImage) → LCM %s", args.ros_rgb_topic, rgb_channel)
    else:
        node.create_subscription(RosImage, args.ros_rgb_topic, _on_rgb_raw, best_effort_qos)
        logger.info("ROS2 %s (Image) → LCM %s", args.ros_rgb_topic, rgb_channel)

    # ── Depth subscriber ──────────────────────────────────────────────────────
    def _on_depth_raw(msg: "RosImage") -> None:
        try:
            data = ros_image_depth_to_lcm_bytes(msg)
            lc.publish(depth_channel, data)
            logger.debug("Depth raw → LCM %s (%d B)", depth_channel, len(data))
        except Exception:
            logger.exception("Depth encode error")

    def _on_depth_compressed(msg: "RosCompressedImage") -> None:
        try:
            data = ros_compressed_image_to_lcm_bytes(msg)
            lc.publish(depth_channel, data)
            logger.debug("Depth compressed → LCM %s (%d B)", depth_channel, len(data))
        except Exception:
            logger.exception("Depth forward error")

    if args.ros_depth_compressed:
        node.create_subscription(RosCompressedImage, args.ros_depth_topic, _on_depth_compressed, best_effort_qos)
        logger.info("ROS2 %s (CompressedImage) → LCM %s", args.ros_depth_topic, depth_channel)
    else:
        node.create_subscription(RosImage, args.ros_depth_topic, _on_depth_raw, best_effort_qos)
        logger.info("ROS2 %s (Image) → LCM %s", args.ros_depth_topic, depth_channel)

    # ── Camera info subscriber ────────────────────────────────────────────────
    if not args.no_camera_info:
        def _on_camera_info(msg: "RosCameraInfo") -> None:
            try:
                data = ros_camera_info_to_lcm_bytes(msg)
                lc.publish(camera_info_channel, data)
                logger.debug(
                    "camera_info %dx%d → LCM %s", msg.width, msg.height, camera_info_channel
                )
            except Exception:
                logger.exception("camera_info encode error")

        node.create_subscription(RosCameraInfo, args.ros_camera_info_topic, _on_camera_info, best_effort_qos)
        logger.info("ROS2 %s → LCM %s", args.ros_camera_info_topic, camera_info_channel)

    # ── spin ──────────────────────────────────────────────────────────────────
    import threading

    def _lcm_handle_loop() -> None:
        while rclpy.ok():
            lc.handle_timeout(10)  # 10 ms

    lcm_thread = threading.Thread(target=_lcm_handle_loop, daemon=True)
    lcm_thread.start()

    try:
        logger.info("Bridge running — Ctrl-C to stop")
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

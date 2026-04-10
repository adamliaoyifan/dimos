# Copyright 2026 Dimensional Inc.
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

"""Bridge ROS2 CompressedImage topics → DimOS Image streams.

Subscribes to ROS2 ``sensor_msgs/CompressedImage`` topics (JPEG colour,
PNG uint16 depth) published by an edge node (e.g. ``compressor.py``),
decompresses them, and re-publishes as DimOS ``Out[Image]`` streams on
pSHM so downstream modules (NavDP navigator, VLN skill, …) can consume
them without knowing about ROS2.

Requires ``rclpy`` and ``sensor_msgs`` in the same Python environment (ROS distro packages
or pip when wheels exist). If they are missing, use ``navdp.trajectory_camera: go2`` to omit
this bridge.

Usage in a blueprint::

    from dimos.hardware.sensors.camera.ros2_compressed_bridge import (
        ROS2CompressedImageBridge,
    )

    bridge = ROS2CompressedImageBridge.blueprint(
        rgb_topic="/camera/camera/color/image_raw/compressed",
        depth_topic="/camera/camera/aligned_depth_to_color/image_raw/compressed",
    )
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from dimos.core.core import rpc
from dimos.core.module import Module, ModuleConfig
from dimos.core.stream import Out
from dimos.hardware.sensors.camera.compressed_image_decode import (
    decode_jpeg_to_rgb_image,
    decode_png_depth_to_image,
    normalize_compressed_payload,
)
from dimos.msgs.sensor_msgs.Image import Image

logger = logging.getLogger(__name__)


class ROS2CompressedBridgeConfig(ModuleConfig):
    """Configuration for the ROS2 compressed-image bridge."""

    rgb_topic: str = "/camera/camera/color/image_raw/compressed"
    depth_topic: str = "/camera/camera/aligned_depth_to_color/image_raw/compressed"
    # Divisor to convert raw uint16 depth to float32 metres.
    # 1000.0 → depth in millimetres (RealSense default).
    depth_scale: float = 1000.0
    # QoS depth for ROS2 subscriptions.
    qos_depth: int = 10


class ROS2CompressedImageBridge(Module[ROS2CompressedBridgeConfig]):
    """Subscribe to ROS2 CompressedImage, decompress, publish as DimOS Image.

    Outputs
    -------
    color_image : Out[Image]
        Decompressed RGB image (``ImageFormat.RGB``).
    depth_image : Out[Image]
        Decompressed depth in float32 metres (``ImageFormat.DEPTH``).
    """

    color_image: Out[Image]
    depth_image: Out[Image]

    default_config = ROS2CompressedBridgeConfig

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._ros_node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._running = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @rpc
    def start(self) -> None:
        try:
            import rclpy
            from rclpy.executors import SingleThreadedExecutor
            from rclpy.node import Node
            from rclpy.qos import (
                QoSDurabilityPolicy,
                QoSHistoryPolicy,
                QoSProfile,
                QoSReliabilityPolicy,
            )
            from sensor_msgs.msg import CompressedImage as ROSCompressedImage
        except ImportError as exc:
            raise ImportError(
                "rclpy and sensor_msgs are required for ROS2CompressedImageBridge. "
                "Install for your ROS distro (e.g. apt ros-<distro>-rclpy and use that Python), "
                "or pip install rclpy sensor-msgs when wheels exist for your platform. "
                "Alternatively set navdp.trajectory_camera: go2 in vln_config.yaml to skip this bridge."
            ) from exc

        if not rclpy.ok():
            rclpy.init()

        self._ros_node = Node("dimos_compressed_bridge")

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=self.config.qos_depth,
        )

        self._ros_node.create_subscription(
            ROSCompressedImage,
            self.config.rgb_topic,
            self._on_compressed_rgb,
            qos,
        )
        self._ros_node.create_subscription(
            ROSCompressedImage,
            self.config.depth_topic,
            self._on_compressed_depth,
            qos,
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(self._ros_node)

        self._running = True
        self._spin_thread = threading.Thread(
            target=self._spin_loop, daemon=True, name="ros2_bridge_spin"
        )
        self._spin_thread.start()

        logger.info(
            "[ROS2Bridge] Subscribed: rgb=%s  depth=%s",
            self.config.rgb_topic,
            self.config.depth_topic,
        )

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._executor is not None:
            self._executor.shutdown()
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=2.0)
            self._spin_thread = None
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
        self._executor = None
        super().stop()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _spin_loop(self) -> None:
        while self._running:
            try:
                if self._executor is not None:
                    self._executor.spin_once(timeout_sec=0.01)
            except Exception:
                break

    def _on_compressed_rgb(self, msg: Any) -> None:
        """Decompress JPEG → RGB DimOS Image."""
        payload = normalize_compressed_payload(msg.data)
        img = decode_jpeg_to_rgb_image(payload, log_prefix="[ROS2Bridge]")
        if img is not None:
            self.color_image.publish(img)

    def _on_compressed_depth(self, msg: Any) -> None:
        """Decompress PNG uint16 → float32 metres DimOS Image."""
        payload = normalize_compressed_payload(msg.data)
        img = decode_png_depth_to_image(
            payload,
            self.config.depth_scale,
            log_prefix="[ROS2Bridge]",
        )
        if img is not None:
            self.depth_image.publish(img)

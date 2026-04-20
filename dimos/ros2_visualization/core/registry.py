# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""BridgeRegistry — central hub that wires adapters to bridges.

Usage::

    registry = BridgeRegistry(node_name="dimos_viz")
    registry.register(OdomBridge())
    registry.register(TrajectoryBridge())
    registry.start()

    # From any adapter thread:
    registry.publish(OdomSample(...))

    # Shutdown:
    registry.stop()
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.frames import FrameIds
from dimos.ros2_visualization.core.schema import (
    OdomSample,
)

logger = logging.getLogger(__name__)


class BridgeRegistry:
    """Owns the single rclpy Node and all registered Bridge instances.

    Thread-safe: ``publish()`` may be called from any thread simultaneously.
    """

    def __init__(
        self,
        node_name: str = "ros2_visualization",
        frames: FrameIds | None = None,
        spin_in_background: bool = True,
    ) -> None:
        self._node_name = node_name
        self.frames = frames or FrameIds()
        self._spin_in_background = spin_in_background
        self._bridges: dict[type, Bridge] = {}
        self._lock = threading.Lock()
        self._node: Any = None
        self._executor: Any = None
        self._spin_thread: threading.Thread | None = None
        self._started = False

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, bridge: Bridge) -> "BridgeRegistry":
        """Register *bridge* keyed by its ``sample_type``.  Returns self for chaining."""
        with self._lock:
            if bridge.sample_type in self._bridges:
                logger.warning(
                    "BridgeRegistry: overwriting existing bridge for %s",
                    bridge.sample_type.__name__,
                )
            self._bridges[bridge.sample_type] = bridge
        return self

    def register_all(self, *bridges: Bridge) -> "BridgeRegistry":
        for b in bridges:
            self.register(b)
        return self

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Initialise rclpy, create the shared Node, and start all bridges."""
        import rclpy  # type: ignore[import-untyped]
        from rclpy.executors import MultiThreadedExecutor  # type: ignore[import-untyped]

        if not rclpy.ok():
            rclpy.init()

        self._node = rclpy.create_node(self._node_name)
        self._executor = MultiThreadedExecutor()
        self._executor.add_node(self._node)

        with self._lock:
            for bridge in self._bridges.values():
                bridge.start(self._node)

        if self._spin_in_background:
            self._spin_thread = threading.Thread(
                target=self._executor.spin,
                daemon=True,
                name="ros2_viz_spin",
            )
            self._spin_thread.start()

        self._started = True
        logger.info("BridgeRegistry started with %d bridges.", len(self._bridges))

    def stop(self) -> None:
        """Shutdown all bridges, the executor, and rclpy."""
        if not self._started:
            return

        with self._lock:
            for bridge in self._bridges.values():
                bridge.stop()

        if self._executor is not None:
            self._executor.shutdown(timeout_sec=2.0)

        if self._spin_thread is not None:
            self._spin_thread.join(timeout=3.0)

        import rclpy  # type: ignore[import-untyped]

        if rclpy.ok():
            rclpy.shutdown()

        self._started = False
        logger.info("BridgeRegistry stopped.")

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    def publish(self, sample: Any) -> None:
        """Forward *sample* to the registered bridge for its type.

        Silently drops the sample if no bridge is registered (avoids crashing
        adapters that publish optional data types).
        """
        bridge = self._bridges.get(type(sample))
        if bridge is None:
            return
        try:
            bridge.on_sample(sample)
        except Exception:
            logger.exception(
                "Bridge %s raised while handling %s",
                bridge.name,
                type(sample).__name__,
            )

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    @property
    def node(self) -> Any:
        """The underlying rclpy Node (available after start())."""
        return self._node

    def registered_types(self) -> list[str]:
        return [t.__name__ for t in self._bridges]

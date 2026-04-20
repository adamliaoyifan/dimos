# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Clock utilities and /clock publisher.

``ClockPublisher`` runs in a background thread and publishes
``rosgraph_msgs/Clock`` at 100 Hz so Foxglove's timeline slider and
``ros2 bag play --clock`` work correctly during replay.

``mono_to_ros_time(ns)`` converts a monotonic-nanosecond stamp (as stored in
DimOS ``Timestamped`` objects via ``time.monotonic_ns()``) to a
``builtin_interfaces/Time`` message.  All bridges must use this function —
never call ``time.time()`` directly in a bridge.
"""

from __future__ import annotations

import time
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # rclpy imported lazily


def now_ns() -> int:
    """Return the current wall-clock time in nanoseconds."""
    return time.time_ns()


def ns_to_sec_nsec(stamp_ns: int) -> tuple[int, int]:
    """Split a nanosecond stamp into (sec, nanosec) for builtin_interfaces/Time."""
    sec = stamp_ns // 1_000_000_000
    nsec = stamp_ns % 1_000_000_000
    return int(sec), int(nsec)


class ClockPublisher:
    """Publishes ``rosgraph_msgs/Clock`` at *rate_hz* so viewers track wall time.

    Must be constructed after ``rclpy.init()`` has been called.
    Call ``start()`` to begin publishing; ``stop()`` to shut down.
    """

    def __init__(self, node: "rclpy.node.Node", rate_hz: float = 100.0) -> None:  # type: ignore[name-defined]
        from rosgraph_msgs.msg import Clock  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._node = node
        self._rate_hz = rate_hz
        self._pub = node.create_publisher(Clock, "/clock", build(QoSProfile.RELIABLE))
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._Clock = Clock

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="clock_publisher")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _loop(self) -> None:
        interval = 1.0 / self._rate_hz
        while not self._stop_event.is_set():
            msg = self._Clock()
            sec, nsec = ns_to_sec_nsec(now_ns())
            msg.clock.sec = sec
            msg.clock.nanosec = nsec
            self._pub.publish(msg)
            time.sleep(interval)

    @staticmethod
    def make_ros_time(stamp_ns: int) -> "builtin_interfaces.msg.Time":  # type: ignore[name-defined]
        """Build a ``builtin_interfaces/Time`` from a nanosecond stamp."""
        from builtin_interfaces.msg import Time  # type: ignore[import-untyped]

        t = Time()
        t.sec, t.nanosec = ns_to_sec_nsec(stamp_ns)
        return t

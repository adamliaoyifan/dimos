# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Pre-tuned ROS 2 QoS profiles.

Imported lazily inside Bridge.start() so the module can be imported without rclpy.
Call ``build()`` after rclpy has been initialised.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class QoSProfile(Enum):
    """Symbolic QoS profile names used by bridges."""

    SENSOR_DATA = "sensor_data"       # best-effort, depth 5  — images, pointclouds
    RELIABLE = "reliable"             # reliable, depth 10    — costmap, path, odom
    RELIABLE_LATCHED = "reliable_latched"  # reliable, depth 1, transient-local — robot geom
    BEST_EFFORT = "best_effort"       # best-effort, depth 1  — high-rate markers


def build(profile: QoSProfile):  # type: ignore[return]
    """Return an rclpy QoSProfile matching *profile*.  Must be called after rclpy.init()."""
    from rclpy.qos import (  # type: ignore[import-untyped]
        DurabilityPolicy,
        HistoryPolicy,
        QoSProfile as RclQoS,
        ReliabilityPolicy,
    )

    match profile:
        case QoSProfile.SENSOR_DATA:
            return RclQoS(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=5,
            )
        case QoSProfile.RELIABLE:
            return RclQoS(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=10,
            )
        case QoSProfile.RELIABLE_LATCHED:
            return RclQoS(
                reliability=ReliabilityPolicy.RELIABLE,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
        case QoSProfile.BEST_EFFORT:
            return RclQoS(
                reliability=ReliabilityPolicy.BEST_EFFORT,
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
            )
        case _:
            raise ValueError(f"Unknown QoS profile: {profile}")

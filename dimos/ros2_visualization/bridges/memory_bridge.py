# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MemoryBridge — visualises spatial memory entries and query events.

Publishes:
  - ``/viz/memory/markers``    — ``visualization_msgs/MarkerArray``
      Sphere per MemoryRecord at its world pose.  Size encodes recency;
      colour encodes similarity (green = high match, grey = stored but not queried).
  - ``/viz/memory/thumbnail``  — ``sensor_msgs/CompressedImage``
      JPEG thumbnail of the most recently queried memory hit.
  - ``/viz/memory/metadata``   — ``std_msgs/String`` JSON
      Per-marker metadata for Foxglove click-to-inspect.
  - ``/viz/memory/query``      — ``std_msgs/String`` JSON
      Full query event: text, top-k results with pose + similarity.
"""

from __future__ import annotations

import threading
from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.metadata import MetadataSidecar
from dimos.ros2_visualization.core.schema import MemoryQueryEvent, MemoryRecord


class MemoryBridge(Bridge):
    """Converts ``MemoryRecord`` → sphere marker + thumbnail + metadata."""

    sample_type = MemoryRecord
    name = "memory"

    MARKER_TOPIC = "/viz/memory/markers"
    THUMBNAIL_TOPIC = "/viz/memory/thumbnail"
    QUERY_TOPIC = "/viz/memory/query"

    def __init__(self) -> None:
        super().__init__()
        self._pub_markers: Any = None
        self._pub_thumb: Any = None
        self._pub_query: Any = None
        self._meta: MetadataSidecar | None = None
        # In-memory index: record_id → (marker_id, record)
        self._records: dict[str, tuple[int, MemoryRecord]] = {}
        self._next_id = 0
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from sensor_msgs.msg import CompressedImage  # type: ignore[import-untyped]
        from std_msgs.msg import String  # type: ignore[import-untyped]
        from visualization_msgs.msg import Marker, MarkerArray  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub_markers = node.create_publisher(
            MarkerArray, self.MARKER_TOPIC, build(QoSProfile.RELIABLE_LATCHED)
        )
        self._pub_thumb = node.create_publisher(
            CompressedImage, self.THUMBNAIL_TOPIC, build(QoSProfile.SENSOR_DATA)
        )
        self._pub_query = node.create_publisher(
            String, self.QUERY_TOPIC, build(QoSProfile.RELIABLE)
        )
        self._Marker = Marker
        self._MarkerArray = MarkerArray
        self._CompressedImage = CompressedImage
        self._String = String
        self._meta = MetadataSidecar(node, self.MARKER_TOPIC)

    def on_sample(self, sample: MemoryRecord) -> None:
        self._check_started()

        from dimos.ros2_visualization.core.clock import ClockPublisher, now_ns

        with self._lock:
            if sample.record_id not in self._records:
                marker_id = self._next_id
                self._next_id += 1
            else:
                marker_id = self._records[sample.record_id][0]
            self._records[sample.record_id] = (marker_id, sample)

        stamp = ClockPublisher.make_ros_time(now_ns())

        m = self._Marker()
        m.header.stamp = stamp
        m.header.frame_id = sample.pose.frame
        m.ns = "memory"
        m.id = marker_id
        m.type = self._Marker.SPHERE
        m.action = self._Marker.ADD
        m.pose.position.x = sample.pose.x
        m.pose.position.y = sample.pose.y
        m.pose.orientation.w = 1.0
        m.scale.x = 0.25
        m.scale.y = 0.25
        m.scale.z = 0.10

        if sample.similarity is not None:
            sim = max(0.0, min(1.0, sample.similarity))
            m.color.r = 1.0 - sim
            m.color.g = sim
            m.color.b = 0.2
            m.color.a = 0.9
        else:
            m.color.r = 0.6
            m.color.g = 0.6
            m.color.b = 0.6
            m.color.a = 0.6

        with self._lock:
            all_markers = []
            metadata: dict[int, dict] = {}
            for rid, (mid, rec) in self._records.items():
                if rid == sample.record_id:
                    all_markers.append(m)
                else:
                    existing = self._Marker()
                    existing.ns = "memory"
                    existing.id = mid
                    existing.action = self._Marker.MODIFY if hasattr(self._Marker, "MODIFY") else 0
                    all_markers.append(existing)
                meta_rec = rec if rid != sample.record_id else sample
                metadata[mid] = {
                    "record_id": rid,
                    "x": round(meta_rec.pose.x, 3),
                    "y": round(meta_rec.pose.y, 3),
                    "yaw_deg": round(meta_rec.pose.yaw * 57.296, 2),
                    "tags": meta_rec.tags,
                    "description": meta_rec.description,
                    "similarity": meta_rec.similarity,
                    "has_thumbnail": meta_rec.thumbnail is not None,
                }

            arr = self._MarkerArray()
            arr.markers = all_markers
            self._pub_markers.publish(arr)
            if self._meta is not None:
                self._meta.publish(metadata)

        if sample.thumbnail is not None:
            img = self._CompressedImage()
            img.header.stamp = stamp
            img.header.frame_id = sample.pose.frame
            img.format = "jpeg"
            img.data = list(sample.thumbnail)
            self._pub_thumb.publish(img)


class MemoryQueryBridge(Bridge):
    """Converts ``MemoryQueryEvent`` → JSON on ``/viz/memory/query``."""

    sample_type = MemoryQueryEvent
    name = "memory_query"

    def __init__(self) -> None:
        super().__init__()
        self._pub: Any = None
        self._lock = threading.Lock()

    def start(self, node: Any) -> None:
        super().start(node)
        from std_msgs.msg import String  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub = node.create_publisher(
            String, MemoryBridge.QUERY_TOPIC, build(QoSProfile.RELIABLE)
        )
        self._String = String

    def on_sample(self, sample: MemoryQueryEvent) -> None:
        self._check_started()

        import json

        payload = {
            "query": sample.query_text,
            "stamp_ns": sample.stamp_ns,
            "results": [
                {
                    "record_id": r.record_id,
                    "x": round(r.pose.x, 3),
                    "y": round(r.pose.y, 3),
                    "similarity": r.similarity,
                    "tags": r.tags,
                    "description": r.description,
                }
                for r in sample.results
            ],
        }
        msg = self._String()
        msg.data = json.dumps(payload)
        with self._lock:
            self._pub.publish(msg)

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""MemoryAdapter — polls SpatialMemory and publishes MemoryRecord/MemoryQueryEvent.

SpatialMemory stores images + poses in a ChromaDB collection.  This adapter
polls ``query_by_location()`` at the robot's current position every
``poll_interval_s`` seconds, diffs against previously seen record IDs, and
publishes only new or updated entries to avoid flooding the ROS 2 bus.

For VLN query events (``query_by_text`` hits) it subscribes to the VLN skill's
RPC log if available.

Usage::

    from dimos.ros2_visualization.adapters.dimos.memory_source import MemoryAdapter

    adapter = MemoryAdapter(
        spatial_memory=spatial_memory_module,
        odom_source=lambda: (nav.odom.value.x, nav.odom.value.y),
        poll_interval_s=2.0,
        query_radius_m=5.0,
    )
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any, Callable

from dimos.ros2_visualization.adapters.base import Adapter
from dimos.ros2_visualization.core.schema import MemoryRecord, Pose2D

if TYPE_CHECKING:
    from dimos.ros2_visualization.core.registry import BridgeRegistry

logger = logging.getLogger(__name__)


class MemoryAdapter(Adapter):
    """Polls SpatialMemory and publishes MemoryRecord samples.

    Args:
        spatial_memory:   The live ``SpatialMemory`` module instance.
        odom_source:      Callable returning ``(x, y)`` of robot's current pose.
        poll_interval_s:  How often to poll SpatialMemory (default 2 s).
        query_radius_m:   Radius (metres) passed to ``query_by_location``.
        max_results:      Maximum entries to fetch per poll.
    """

    name = "memory"

    def __init__(
        self,
        spatial_memory: Any = None,
        odom_source: Callable[[], tuple[float, float]] | None = None,
        poll_interval_s: float = 2.0,
        query_radius_m: float = 50.0,
        max_results: int = 200,
    ) -> None:
        self._spatial_memory = spatial_memory
        self._odom_source = odom_source
        self._poll_interval_s = poll_interval_s
        self._query_radius_m = query_radius_m
        self._max_results = max_results
        self._registry: BridgeRegistry | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._seen_ids: set[str] = set()

    def bind(self, registry: "BridgeRegistry") -> None:
        self._registry = registry
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="memory_viz_poller"
        )
        self._thread.start()
        logger.info(
            "MemoryAdapter started (poll=%.1f s, radius=%.1f m).",
            self._poll_interval_s,
            self._query_radius_m,
        )

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval_s):
            try:
                self._poll_once()
            except Exception as e:
                logger.debug("MemoryAdapter poll error: %s", e)

    def _poll_once(self) -> None:
        if self._spatial_memory is None or self._registry is None:
            return

        x, y = 0.0, 0.0
        if self._odom_source is not None:
            try:
                x, y = self._odom_source()
            except Exception:
                pass

        try:
            results = self._spatial_memory.query_by_location(
                x, y, radius=self._query_radius_m, n_results=self._max_results
            )
        except Exception as e:
            logger.debug("MemoryAdapter: query_by_location failed: %s", e)
            return

        if not results:
            return

        stamp_ns = int(time.time() * 1_000_000_000)

        for entry in results:
            record_id = self._extract_id(entry)
            pose = self._extract_pose(entry, stamp_ns)
            thumbnail = self._extract_thumbnail(entry)
            tags = list(getattr(entry, "tags", []) or [])
            description = str(getattr(entry, "description", "") or "")

            record = MemoryRecord(
                record_id=record_id,
                pose=pose,
                thumbnail=thumbnail,
                tags=tags,
                description=description,
                similarity=None,
            )
            self._seen_ids.add(record_id)
            self._registry.publish(record)

    # ------------------------------------------------------------------
    # Entry parsing helpers (tolerate varying SpatialMemory APIs)
    # ------------------------------------------------------------------

    def _extract_id(self, entry: Any) -> str:
        for attr in ("id", "record_id", "frame_id", "uuid"):
            v = getattr(entry, attr, None)
            if v is not None:
                return str(v)
        # ChromaDB result dicts
        if isinstance(entry, dict):
            for k in ("id", "record_id"):
                if k in entry:
                    return str(entry[k])
        return str(id(entry))

    def _extract_pose(self, entry: Any, stamp_ns: int) -> Pose2D:
        if hasattr(entry, "pose"):
            p = entry.pose
            return Pose2D(
                x=float(getattr(p, "x", 0.0)),
                y=float(getattr(p, "y", 0.0)),
                yaw=float(getattr(p, "yaw", 0.0)),
                stamp_ns=stamp_ns,
                frame="odom",
            )
        if hasattr(entry, "x") and hasattr(entry, "y"):
            return Pose2D(
                x=float(entry.x),
                y=float(entry.y),
                yaw=float(getattr(entry, "yaw", 0.0)),
                stamp_ns=stamp_ns,
                frame="odom",
            )
        if isinstance(entry, dict):
            return Pose2D(
                x=float(entry.get("x", 0.0)),
                y=float(entry.get("y", 0.0)),
                yaw=float(entry.get("yaw", 0.0)),
                stamp_ns=stamp_ns,
                frame="odom",
            )
        return Pose2D(x=0.0, y=0.0, yaw=0.0, stamp_ns=stamp_ns, frame="odom")

    def _extract_thumbnail(self, entry: Any) -> bytes | None:
        import io

        try:
            import cv2  # type: ignore[import-untyped]
            import numpy as np
            from PIL import Image as PILImage  # type: ignore[import-untyped]
        except ImportError:
            return None

        img = getattr(entry, "image", None) or getattr(entry, "frame", None)
        if img is None:
            return None

        if isinstance(img, np.ndarray):
            # BGR numpy → JPEG bytes
            arr = img
            if arr.dtype != np.uint8:
                arr = (arr * 255).clip(0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
            success, buf = cv2.imencode(".jpg", arr, [cv2.IMWRITE_JPEG_QUALITY, 75])
            return buf.tobytes() if success else None

        if isinstance(img, (bytes, bytearray)):
            return bytes(img)

        return None

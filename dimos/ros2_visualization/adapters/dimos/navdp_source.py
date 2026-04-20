# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""NavDPAdapter — polls NavDP navigator for trajectory candidates.

The NavDP inference thread stores ``latest_traj`` as an attribute on the
navigator object (not an RxPY stream).  This adapter polls it at
``poll_hz`` and publishes each new trajectory as a ``TrajSample``.

Usage::

    from dimos.ros2_visualization.adapters.dimos.navdp_source import NavDPAdapter

    adapter = NavDPAdapter(
        navigator=nav_module,   # NavDPNavigator instance
        poll_hz=10.0,
    )
"""

from __future__ import annotations

import logging
import threading
import time
from typing import TYPE_CHECKING, Any

from dimos.ros2_visualization.adapters.base import Adapter
from dimos.ros2_visualization.core.schema import Pose2D, TrajSample, WaypointMeta

if TYPE_CHECKING:
    from dimos.ros2_visualization.core.registry import BridgeRegistry

logger = logging.getLogger(__name__)


class NavDPAdapter(Adapter):
    """Polls NavDPNavigator._latest_traj and publishes ``TrajSample`` to registry.

    Args:
        navigator:    The live ``NavDPNavigator`` module instance.
        poll_hz:      How often to poll for a new trajectory (default 10 Hz).
        odom_source:  Optional callable returning ``(x, y, yaw)`` for current pose,
                      used to compute relative trajectory frames.
    """

    name = "navdp"

    def __init__(
        self,
        navigator: Any = None,
        poll_hz: float = 10.0,
        odom_source: Any = None,
    ) -> None:
        self._navigator = navigator
        self._poll_interval = 1.0 / max(1.0, poll_hz)
        self._odom_source = odom_source
        self._registry: BridgeRegistry | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_traj_hash: int = 0

    def bind(self, registry: "BridgeRegistry") -> None:
        self._registry = registry
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="navdp_viz_poller"
        )
        self._thread.start()
        logger.info("NavDPAdapter started polling at %.1f Hz.", 1.0 / self._poll_interval)

    def shutdown(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _poll_loop(self) -> None:
        while not self._stop_event.wait(self._poll_interval):
            try:
                self._poll_once()
            except Exception as e:
                logger.debug("NavDPAdapter poll error: %s", e)

    def _poll_once(self) -> None:
        if self._navigator is None or self._registry is None:
            return

        # NavDPNavigator exposes _traj_selector._latest_selected and the full
        # candidate list via the trajectory selector.
        traj_selector = getattr(self._navigator, "_traj_selector", None)
        if traj_selector is None:
            return

        candidates = getattr(traj_selector, "_candidates", None) or []
        selected_idx = getattr(traj_selector, "_selected_idx", 0)

        for traj_idx, candidate in enumerate(candidates):
            traj_hash = hash((traj_idx, id(candidate)))
            if traj_hash == self._last_traj_hash and traj_idx == 0:
                return  # nothing changed
            self._last_traj_hash = traj_hash

            points_raw = getattr(candidate, "poses", None) or getattr(candidate, "points", None) or []
            costs_raw = getattr(candidate, "costs", None) or []
            speeds_raw = getattr(candidate, "speeds", None) or []

            stamp_ns = int(time.time() * 1_000_000_000)
            frame = "odom"

            pts: list[Pose2D] = []
            metas: list[WaypointMeta] = []

            for i, pt in enumerate(points_raw):
                if hasattr(pt, "x"):
                    x, y, yaw = float(pt.x), float(pt.y), float(getattr(pt, "yaw", 0.0))
                elif hasattr(pt, "__len__") and len(pt) >= 2:
                    x, y = float(pt[0]), float(pt[1])
                    yaw = float(pt[2]) if len(pt) > 2 else 0.0
                else:
                    continue

                pts.append(Pose2D(x=x, y=y, yaw=yaw, stamp_ns=stamp_ns, frame=frame))
                critic = float(costs_raw[i]) if i < len(costs_raw) else 0.0
                speed = float(speeds_raw[i]) if i < len(speeds_raw) else 0.0
                metas.append(WaypointMeta(critic_score=critic, speed=speed))

            if not pts:
                continue

            is_selected = traj_idx == selected_idx
            color = (0.0, 1.0, 0.3) if is_selected else (0.8, 0.8, 0.2)

            sample = TrajSample(
                traj_id=f"navdp_{traj_idx}",
                points=pts,
                waypoint_meta=metas,
                is_selected=is_selected,
                color_rgb=color,
                stamp_ns=stamp_ns,
                frame=frame,
                source="navdp",
            )
            self._registry.publish(sample)

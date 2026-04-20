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

"""Room exit planner for memory-augmented frontier-based navigation.

When the robot has exhausted frontiers in a room (room saturation), this
helper computes the best backtrack waypoint along the entry trail that leads
toward unexplored space — likely a corridor or doorway the robot entered from.

Design notes:
- Pure Python helper (no Module/streams) for testability.
- Depends only on the raw explore trail and SpatialMemory RPC callable.
- Does NOT navigate by itself; callers are responsible for sending the
  returned waypoint to the navigator.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from dimos.utils.logging_config import setup_logger

logger = setup_logger()


@dataclass
class RoomExitConfig:
    """Tuning parameters for the room exit planner."""

    trail_sample_spacing_m: float = 1.0
    """Minimum spacing (metres) between thinned trail waypoints."""

    backtrack_novelty_threshold: float = 0.6
    """Stop backtracking when memory novelty score exceeds this (0–1).
    Higher = less explored area required before committing to a waypoint."""

    memory_query_radius_m: float = 1.5
    """Radius (metres) for SpatialMemory ``query_by_location`` lookups."""

    semantic_bias_enabled: bool = True
    """Query SpatialMemory for hallway/corridor images to bias the target."""

    min_trail_points: int = 3
    """Minimum thinned trail points required to attempt backtracking."""


@dataclass
class BacktrackTarget:
    """Result of ``RoomExitPlanner.find_backtrack_target``."""

    x: float
    y: float
    novelty_score: float
    """0 = fully explored near this point; 1 = completely novel."""
    trail_index: int
    """Index in the thinned trail (from the reversed/entry-end perspective)."""
    reason: str = ""


class RoomExitPlanner:
    """Computes a backtrack waypoint toward the room entry corridor.

    Usage::

        planner = RoomExitPlanner(
            config=RoomExitConfig(),
            query_by_location_fn=spatial_memory.query_by_location,
            query_by_text_fn=spatial_memory.query_by_text,
        )
        raw_trail = navigator.get_explore_trail()
        target = planner.find_backtrack_target(raw_trail)
        if target:
            navigate_to(target.x, target.y)
    """

    def __init__(
        self,
        config: RoomExitConfig | None = None,
        query_by_location_fn: Callable[..., list[dict[str, Any]]] | None = None,
        query_by_text_fn: Callable[..., list[dict[str, Any]]] | None = None,
    ) -> None:
        self.config = config or RoomExitConfig()
        self._query_by_location = query_by_location_fn
        self._query_by_text = query_by_text_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def thin_trail(
        self, trail: list[tuple[float, float]]
    ) -> list[tuple[float, float]]:
        """Subsample the trail to at most one point per ``trail_sample_spacing_m``.

        Args:
            trail: Raw (x, y) trail in chronological order (oldest first).

        Returns:
            Thinned trail in the same chronological order.
        """
        if not trail:
            return []

        thinned: list[tuple[float, float]] = [trail[0]]
        for pt in trail[1:]:
            px, py = thinned[-1]
            if math.hypot(pt[0] - px, pt[1] - py) >= self.config.trail_sample_spacing_m:
                thinned.append(pt)

        return thinned

    def compute_novelty_score(self, x: float, y: float) -> float:
        """Return a novelty score [0, 1] for a world position.

        A high score means few past observations are nearby → likely unexplored
        corridor/exit.  Returns 0.5 when SpatialMemory is unavailable.

        Args:
            x: World-frame x position (metres).
            y: World-frame y position (metres).

        Returns:
            Novelty score in [0, 1].
        """
        if self._query_by_location is None:
            return 0.5

        try:
            nearby = self._query_by_location(
                x, y, self.config.memory_query_radius_m, 20
            )
            n = len(nearby)
            return 1.0 / (1.0 + n)
        except Exception as exc:
            logger.warning("RoomExitPlanner: query_by_location failed: %s", exc)
            return 0.5

    def _semantic_bias_positions(self) -> list[tuple[float, float]]:
        """Return (x, y) positions of hallway/corridor images in SpatialMemory.

        Used to optionally bias the backtrack target toward semantically
        matched locations.  Returns empty list if query fails or is disabled.
        """
        if not self.config.semantic_bias_enabled or self._query_by_text is None:
            return []

        try:
            results = self._query_by_text(
                "hallway OR corridor OR door OR entrance OR exit", limit=10
            )
            positions: list[tuple[float, float]] = []
            for r in results:
                meta = r.get("metadata", {})
                px = meta.get("pos_x")
                py = meta.get("pos_y")
                if px is not None and py is not None:
                    positions.append((float(px), float(py)))
            return positions
        except Exception as exc:
            logger.warning("RoomExitPlanner: semantic query failed: %s", exc)
            return []

    def find_backtrack_target(
        self, raw_trail: list[tuple[float, float]]
    ) -> BacktrackTarget | None:
        """Walk the entry trail in reverse to find the most novel exit waypoint.

        Strategy:
        1. Thin the trail to ~``trail_sample_spacing_m`` intervals.
        2. Walk the thinned trail from newest → oldest (reverse = entry direction).
        3. At each waypoint compute a novelty score via SpatialMemory.
        4. Return the first point where novelty >= threshold, OR the point with
           the highest novelty if none exceeds the threshold.
        5. Optionally bias toward semantically matched hallway poses.

        Args:
            raw_trail: Raw explore trail from ``NavDPNavigator.get_explore_trail()``.

        Returns:
            BacktrackTarget with the recommended exit waypoint, or None if the
            trail is too short to reason about.
        """
        thinned = self.thin_trail(raw_trail)
        if len(thinned) < self.config.min_trail_points:
            logger.info(
                "RoomExitPlanner: trail too short (%d thinned points, need %d)",
                len(thinned),
                self.config.min_trail_points,
            )
            return None

        semantic_poses = self._semantic_bias_positions()

        best_target: BacktrackTarget | None = None
        best_score = -1.0

        # Walk from newest entry backward (index len-1 → 0).
        # We skip the last ~2 points (current position area) to avoid picking
        # a waypoint that's already inside the saturated room.
        reversed_indices = list(range(len(thinned) - 1, -1, -1))
        skip_recent = min(2, len(reversed_indices) // 4)

        for rank, idx in enumerate(reversed_indices):
            if rank < skip_recent:
                continue

            x, y = thinned[idx]
            novelty = self.compute_novelty_score(x, y)

            # Semantic bias: boost novelty if near a hallway/corridor image
            if semantic_poses:
                nearest_semantic_dist = min(
                    math.hypot(x - sx, y - sy) for sx, sy in semantic_poses
                )
                if nearest_semantic_dist < self.config.memory_query_radius_m * 2:
                    novelty = min(1.0, novelty + 0.15)

            candidate = BacktrackTarget(
                x=x,
                y=y,
                novelty_score=novelty,
                trail_index=idx,
                reason=f"trail_idx={idx}, novelty={novelty:.2f}",
            )

            if novelty >= self.config.backtrack_novelty_threshold:
                logger.info(
                    "RoomExitPlanner: found high-novelty backtrack target "
                    "(%.2f, %.2f) novelty=%.2f at trail_idx=%d",
                    x, y, novelty, idx,
                )
                return candidate

            if novelty > best_score:
                best_score = novelty
                best_target = candidate

        if best_target is not None:
            logger.info(
                "RoomExitPlanner: no point above threshold %.2f; "
                "using best available (%.2f, %.2f) novelty=%.2f",
                self.config.backtrack_novelty_threshold,
                best_target.x, best_target.y, best_target.novelty_score,
            )

        return best_target

    def is_near_searched_area(
        self,
        x: float,
        y: float,
        searched_locations: list[tuple[float, float]],
        radius_m: float = 2.0,
    ) -> bool:
        """Return True if (x, y) is within ``radius_m`` of any searched location.

        Used by the anti-re-entry check in ``vln_skill.py`` before navigating
        to a semantic room destination.

        Args:
            x: Query x position (metres).
            y: Query y position (metres).
            searched_locations: List of (x, y) positions already searched.
            radius_m: Exclusion radius.

        Returns:
            True if the position should be skipped (already searched).
        """
        for sx, sy in searched_locations:
            if math.hypot(x - sx, y - sy) < radius_m:
                return True
        return False

    @staticmethod
    def make_searched_tag_name(timestamp: float | None = None) -> str:
        """Return a canonical tag name for a searched room location.

        Args:
            timestamp: Optional Unix timestamp; uses ``time.time()`` if None.

        Returns:
            Tag name string compatible with ``SpatialMemory.tag_location``.
        """
        ts = int(timestamp if timestamp is not None else time.time())
        return f"searched:room:{ts}"

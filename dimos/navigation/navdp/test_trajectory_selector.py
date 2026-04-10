"""Unit tests for TrajectorySelector — costmap collision, depth obstacle, exploration cost.

Tests three key features:
    Fix A:  Out-of-bounds costmap waypoints treated as collisions.
    Fix B:  Depth-based forward obstacle detection.
    Feature: Exploration cost penalises revisiting explored areas during SEEK.

Each test creates a small costmap and/or synthetic data so it runs in <1 s
with no external servers or ROS dependencies.
"""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from dimos.msgs.geometry_msgs.Pose import Pose
from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid
from dimos.navigation.navdp.trajectory_selector import (
    SelectionResult,
    TrajectorySelector,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_costmap(
    width: int = 40,
    height: int = 40,
    resolution: float = 0.05,
    origin_x: float = -1.0,
    origin_y: float = -1.0,
    fill: int = CostValues.FREE,
) -> OccupancyGrid:
    """Create a simple costmap with uniform fill."""
    grid = np.full((height, width), fill, dtype=np.int8)
    origin = Pose()
    origin.position.x = origin_x
    origin.position.y = origin_y
    origin.orientation.w = 1.0
    return OccupancyGrid(
        grid=grid, resolution=resolution, origin=origin, frame_id="map", ts=time.time()
    )


def _make_costmap_with_wall(
    width: int = 40,
    height: int = 40,
    resolution: float = 0.05,
    origin_x: float = -1.0,
    origin_y: float = -1.0,
    wall_col_start: int = 30,
    wall_col_end: int = 40,
) -> OccupancyGrid:
    """Create a costmap with free space and a wall (occupied band on the right)."""
    grid = np.full((height, width), CostValues.FREE, dtype=np.int8)
    grid[:, wall_col_start:wall_col_end] = CostValues.OCCUPIED
    origin = Pose()
    origin.position.x = origin_x
    origin.position.y = origin_y
    origin.orientation.w = 1.0
    return OccupancyGrid(
        grid=grid, resolution=resolution, origin=origin, frame_id="map", ts=time.time()
    )


def _identity_waypoint_fn(traj: np.ndarray) -> np.ndarray:
    """Passthrough: treat camera-frame (T, 3) as base_link (T, 2) — just drop z."""
    return traj[:, :2].copy()


def _make_straight_trajectory(
    dx: float, dy: float, n_points: int = 20
) -> np.ndarray:
    """Return a (T, 3) trajectory going from origin to (dx, dy, 0)."""
    xs = np.linspace(0.0, dx, n_points)
    ys = np.linspace(0.0, dy, n_points)
    zs = np.zeros(n_points)
    return np.stack([xs, ys, zs], axis=1).astype(np.float32)


def _make_candidates(
    trajectories: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Bundle trajectories into (K, T, 3) and uniform critic values (K,)."""
    all_traj = np.stack(trajectories, axis=0)
    all_vals = np.ones(len(trajectories), dtype=np.float32)
    return all_traj, all_vals


# ---------------------------------------------------------------------------
# Fix A: Out-of-bounds = collision
# ---------------------------------------------------------------------------


class TestOutOfBoundsCollision:
    """Waypoints outside the costmap grid must be treated as collisions."""

    def _small_costmap(self) -> OccupancyGrid:
        """1m x 1m grid centred near origin (origin at -0.5, -0.5)."""
        return _make_costmap(
            width=20, height=20, resolution=0.05,
            origin_x=-0.5, origin_y=-0.5, fill=CostValues.FREE,
        )

    def test_trajectory_inside_grid_is_free(self):
        """A trajectory that stays within the grid should not be flagged."""
        costmap = self._small_costmap()
        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=0.8, sample_step=2,
        )
        # Trajectory stays within the 1m x 1m grid (origin at -0.5, robot at 0,0)
        traj_inside = _make_straight_trajectory(dx=0.3, dy=0.0, n_points=10)
        all_traj, all_vals = _make_candidates([traj_inside])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_inside,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )
        assert not result.collision_mask[0], "Trajectory inside grid should be collision-free"
        assert not result.fallback_used

    def test_trajectory_outside_grid_collides(self):
        """A trajectory going far beyond the costmap bounds must collide."""
        costmap = self._small_costmap()
        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=5.0, sample_step=2,
        )
        # Trajectory goes 3m forward — way outside the 1m grid
        traj_outside = _make_straight_trajectory(dx=3.0, dy=0.0, n_points=20)
        all_traj, all_vals = _make_candidates([traj_outside])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_outside,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )
        assert result.collision_mask[0], "Trajectory outside grid must collide"

    def test_prefers_inside_over_outside(self):
        """Given one inside and one outside trajectory, selector picks inside."""
        costmap = self._small_costmap()
        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=5.0, sample_step=2,
            max_trajectory_cost=None,
        )
        traj_inside = _make_straight_trajectory(dx=0.3, dy=0.0)
        traj_outside = _make_straight_trajectory(dx=3.0, dy=0.0)
        all_traj, all_vals = _make_candidates([traj_inside, traj_outside])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_inside,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )
        assert result.index == 0, "Should select the inside trajectory"
        assert not result.collision_mask[0]
        assert result.collision_mask[1]

    def test_all_outside_triggers_fallback(self):
        """When every trajectory goes off-map, fallback_used should be True."""
        costmap = self._small_costmap()
        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=5.0, sample_step=2,
        )
        traj_a = _make_straight_trajectory(dx=3.0, dy=0.0)
        traj_b = _make_straight_trajectory(dx=-3.0, dy=0.0)
        traj_c = _make_straight_trajectory(dx=0.0, dy=3.0)
        all_traj, all_vals = _make_candidates([traj_a, traj_b, traj_c])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_a,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )
        assert result.fallback_used, "All off-map should trigger fallback"
        assert all(result.collision_mask), "Every trajectory should collide"


# ---------------------------------------------------------------------------
# Costmap wall collision (pre-existing feature, regression guard)
# ---------------------------------------------------------------------------


class TestCostmapWallCollision:
    """Trajectories heading into occupied cells must collide."""

    def test_trajectory_into_wall_collides(self):
        costmap = _make_costmap_with_wall(
            width=40, height=40, resolution=0.05,
            origin_x=-1.0, origin_y=-1.0,
            wall_col_start=30, wall_col_end=40,
        )
        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=1.5, sample_step=2,
        )
        # Wall starts at x = origin_x + 30*0.05 = -1.0 + 1.5 = 0.5
        # Robot at (0, 0) heading forward 1.0m → endpoint at (1.0, 0) → hits wall at 0.5
        traj_into_wall = _make_straight_trajectory(dx=1.0, dy=0.0, n_points=20)
        all_traj, all_vals = _make_candidates([traj_into_wall])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_into_wall,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )
        assert result.collision_mask[0], "Trajectory into wall must collide"

    def test_trajectory_parallel_to_wall_is_free(self):
        costmap = _make_costmap_with_wall(
            width=40, height=40, resolution=0.05,
            origin_x=-1.0, origin_y=-1.0,
            wall_col_start=30, wall_col_end=40,
        )
        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=1.0, sample_step=2,
        )
        # Trajectory goes sideways (y direction), stays in free space
        traj_parallel = _make_straight_trajectory(dx=0.0, dy=0.5, n_points=20)
        all_traj, all_vals = _make_candidates([traj_parallel])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_parallel,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )
        assert not result.collision_mask[0], "Sideways trajectory should be free"


# ---------------------------------------------------------------------------
# Fix B: Depth-based forward obstacle detection
# ---------------------------------------------------------------------------


class TestDepthForwardObstacle:
    """Depth image showing a close obstacle must penalise forward trajectories."""

    @staticmethod
    def _close_depth_image(min_depth: float = 0.3, h: int = 480, w: int = 640) -> np.ndarray:
        """Depth image with a close obstacle in the centre."""
        img = np.full((h, w), 2.0, dtype=np.float32)  # everything far
        # Place close obstacle in central strip
        y0, y1 = int(h * 0.3), int(h * 0.7)
        x0, x1 = int(w * 0.3), int(w * 0.7)
        img[y0:y1, x0:x1] = min_depth
        return img

    @staticmethod
    def _far_depth_image(h: int = 480, w: int = 640) -> np.ndarray:
        """Depth image with no close obstacles anywhere."""
        return np.full((h, w), 3.0, dtype=np.float32)

    def test_close_obstacle_penalises_forward_trajectory(self):
        selector = TrajectorySelector(
            enabled=True, depth_obstacle_m=0.5, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
        )
        depth = self._close_depth_image(min_depth=0.2)
        traj_forward = _make_straight_trajectory(dx=1.0, dy=0.0, n_points=20)

        cost, collides = selector._depth_forward_cost(depth, traj_forward[:, :2])
        assert cost > 0, "Close obstacle should add cost for forward trajectory"
        # severity = 1 - 0.2/0.5 = 0.6, forward_fraction ~= min(1.0/1.5, 1) ~ 0.67
        # collides when severity > 0.7 — here severity=0.6, so no hard collision
        assert not collides, "Severity 0.6 should not be hard collision"

    def test_very_close_obstacle_triggers_collision(self):
        selector = TrajectorySelector(
            enabled=True, depth_obstacle_m=0.5, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
        )
        depth = self._close_depth_image(min_depth=0.1)
        traj_forward = _make_straight_trajectory(dx=1.0, dy=0.0, n_points=20)

        cost, collides = selector._depth_forward_cost(depth, traj_forward[:, :2])
        assert cost > 0
        # severity = 1 - 0.1/0.5 = 0.8 > 0.7, forward_fraction ~ 0.67 > 0.3
        assert collides, "Very close obstacle (0.1m) should trigger collision"

    def test_close_obstacle_does_not_penalise_backward_trajectory(self):
        selector = TrajectorySelector(
            enabled=True, depth_obstacle_m=0.5, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
        )
        depth = self._close_depth_image(min_depth=0.2)
        # Trajectory goes backward (negative x in base_link)
        traj_backward = _make_straight_trajectory(dx=-0.5, dy=0.3, n_points=20)

        cost, collides = selector._depth_forward_cost(depth, traj_backward[:, :2])
        assert cost == 0.0, "Backward trajectory should have zero depth cost"
        assert not collides

    def test_far_depth_no_penalty(self):
        selector = TrajectorySelector(
            enabled=True, depth_obstacle_m=0.5, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
        )
        depth = self._far_depth_image()
        traj_forward = _make_straight_trajectory(dx=1.0, dy=0.0, n_points=20)

        cost, collides = selector._depth_forward_cost(depth, traj_forward[:, :2])
        assert cost == 0.0, "Far depth should produce zero cost"
        assert not collides

    def test_depth_disabled_when_threshold_zero(self):
        selector = TrajectorySelector(
            enabled=True, depth_obstacle_m=0.0, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
        )
        depth = self._close_depth_image(min_depth=0.1)
        traj_forward = _make_straight_trajectory(dx=1.0, dy=0.0, n_points=20)
        all_traj, all_vals = _make_candidates([traj_forward])
        odom = (0.0, 0.0, 0.0)

        # Full select() — depth_obstacle_m=0 should skip depth check entirely
        result = selector.select(
            selected_traj=traj_forward,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=None,
            depth_image=depth,
        )
        # Without costmap and with depth disabled, nothing should trigger collision
        assert not result.collision_mask[0]

    def test_depth_integrated_in_select(self):
        """Full select() path: depth obstacle should penalise forward trajectory."""
        selector = TrajectorySelector(
            enabled=True, depth_obstacle_m=0.5, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
            max_trajectory_cost=None,
        )
        depth = self._close_depth_image(min_depth=0.1)
        traj_forward = _make_straight_trajectory(dx=1.0, dy=0.0, n_points=20)
        traj_sideways = _make_straight_trajectory(dx=-0.1, dy=0.5, n_points=20)
        all_traj, all_vals = _make_candidates([traj_forward, traj_sideways])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_forward,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=None,
            depth_image=depth,
        )
        # Forward trajectory should have much higher cost than sideways
        assert result.costs[0] > result.costs[1], (
            "Forward trajectory should cost more with close obstacle"
        )
        assert result.index == 1, "Should select sideways trajectory"


# ---------------------------------------------------------------------------
# Exploration cost
# ---------------------------------------------------------------------------


class TestExplorationCost:
    """Exploration cost should penalise trajectories near explored positions."""

    def _selector(self, **kwargs) -> TrajectorySelector:
        defaults = dict(
            enabled=True,
            explore_weight=5.0,
            explore_radius=2.0,
            explore_endpoint_bonus=1.0,
            costmap_weight=0.8,
            critic_weight=0.2,
            collision_penalty=1000.0,
            max_trajectory_cost=None,
            # Disable recency damping in tests that exercise the full trail.
            recency_damping_count=0,
        )
        defaults.update(kwargs)
        return TrajectorySelector(**defaults)

    def test_no_explored_positions_zero_cost(self):
        """With no explored positions, exploration cost should be zero."""
        selector = self._selector()
        wps = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
        explored = np.zeros((0, 2), dtype=np.float32)
        # Should not crash and should return 0
        cost = selector._exploration_cost(wps, explored)
        assert cost == 0.0

    def test_trajectory_through_explored_area_has_positive_cost(self):
        """Trajectory passing through explored points should be penalised."""
        selector = self._selector(explore_radius=1.0)
        # Explored positions along x-axis
        explored = np.array([
            [0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0],
        ], dtype=np.float32)
        # Trajectory goes along the same x-axis path
        wps = np.array([
            [0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0],
        ], dtype=np.float32)
        cost = selector._exploration_cost(wps, explored)
        # Waypoints are at distance 0 from explored → penalty = 1.0 each
        # Endpoint distance = 0 → novelty bonus = 0
        # Expected: mean(1.0) - 0.0 = 1.0
        assert cost > 0.5, f"Expected high positive cost, got {cost}"

    def test_trajectory_into_novel_area_has_negative_cost(self):
        """Trajectory heading away from explored area should be rewarded."""
        selector = self._selector(explore_radius=1.0, explore_endpoint_bonus=1.0)
        # Explored positions are behind the robot (negative x)
        explored = np.array([
            [-2.0, 0.0], [-1.5, 0.0], [-1.0, 0.0],
        ], dtype=np.float32)
        # Trajectory goes forward into unexplored territory
        wps = np.array([
            [0.0, 0.0], [0.5, 0.0], [1.0, 0.0], [1.5, 0.0],
        ], dtype=np.float32)
        cost = selector._exploration_cost(wps, explored)
        # All waypoints are > 1.0m from explored → penalties = 0
        # Endpoint min_dist ~ 3.5 → clamped to 1.0 → bonus = 1.0
        # Expected: 0.0 - 1.0 = -1.0
        assert cost < 0, f"Expected negative cost (reward), got {cost}"

    def test_exploration_cost_not_applied_when_not_seeking(self):
        """Exploration cost only active during SEEK; APPROACH should ignore it."""
        selector = self._selector(explore_radius=1.0)
        explored = np.array([
            [0.0, 0.0], [0.3, 0.0], [0.6, 0.0],
        ], dtype=np.float32)
        traj = _make_straight_trajectory(dx=0.6, dy=0.0, n_points=10)
        all_traj, all_vals = _make_candidates([traj])
        odom = (0.0, 0.0, 0.0)

        # is_seeking=False → exploration cost should not apply
        result_no_seek = selector.select(
            selected_traj=traj,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=None,
            explored_positions=explored,
            is_seeking=False,
        )

        # is_seeking=True → exploration cost should increase total cost
        result_seek = selector.select(
            selected_traj=traj,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=None,
            explored_positions=explored,
            is_seeking=True,
        )

        assert result_seek.costs[0] > result_no_seek.costs[0], (
            "SEEK mode should add exploration cost, making total higher"
        )

    def test_selector_prefers_novel_trajectory_during_seek(self):
        """During SEEK, selector should prefer trajectory heading into novel area."""
        selector = self._selector(explore_radius=1.5, explore_weight=10.0)
        explored = np.array([
            [0.0, 0.0], [0.3, 0.0], [0.6, 0.0], [0.9, 0.0], [1.2, 0.0],
        ], dtype=np.float32)
        # Trajectory A: revisits explored x-axis
        traj_revisit = _make_straight_trajectory(dx=1.0, dy=0.0, n_points=10)
        # Trajectory B: heads into novel territory (perpendicular)
        traj_novel = _make_straight_trajectory(dx=0.0, dy=1.0, n_points=10)
        all_traj, all_vals = _make_candidates([traj_revisit, traj_novel])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_revisit,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=None,
            explored_positions=explored,
            is_seeking=True,
        )
        assert result.index == 1, (
            f"Should prefer novel trajectory (idx 1), got idx {result.index}"
        )
        assert result.costs[0] > result.costs[1], (
            "Revisit trajectory should have higher cost than novel"
        )


# ---------------------------------------------------------------------------
# Selector disabled / passthrough
# ---------------------------------------------------------------------------


class TestSelectorPassthrough:
    """When disabled, the selector returns the original trajectory unchanged."""

    def test_disabled_returns_original(self):
        selector = TrajectorySelector(enabled=False)
        traj = _make_straight_trajectory(dx=1.0, dy=0.0)
        all_traj, all_vals = _make_candidates([traj])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=None,
        )
        assert result.index == -1
        assert not result.fallback_used
        np.testing.assert_array_equal(result.trajectory, traj)


# ---------------------------------------------------------------------------
# Integration: costmap + depth combined
# ---------------------------------------------------------------------------


class TestCombinedCostmapAndDepth:
    """When both costmap and depth are available, both contribute to the cost."""

    def test_costmap_free_but_depth_close_penalises(self):
        """Costmap shows free but depth camera sees a close wall."""
        costmap = _make_costmap(
            width=100, height=100, resolution=0.05,
            origin_x=-2.5, origin_y=-2.5, fill=CostValues.FREE,
        )
        selector = TrajectorySelector(
            enabled=True, depth_obstacle_m=0.5, collision_penalty=1000.0,
            costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=1.5, sample_step=2,
            max_trajectory_cost=None,
        )
        # Depth shows close obstacle (0.15m) — very close wall
        depth = np.full((480, 640), 2.0, dtype=np.float32)
        depth[100:380, 200:440] = 0.15

        traj_forward = _make_straight_trajectory(dx=1.0, dy=0.0, n_points=20)
        traj_turn = _make_straight_trajectory(dx=-0.1, dy=0.8, n_points=20)
        all_traj, all_vals = _make_candidates([traj_forward, traj_turn])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_forward,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
            depth_image=depth,
        )
        # Even though costmap is all FREE, depth should penalise forward
        assert result.costs[0] > result.costs[1], (
            "Forward trajectory should cost more due to depth obstacle"
        )
        assert result.index == 1, "Should prefer turning away"


# ---------------------------------------------------------------------------
# UNKNOWN cell handling in costmap OBB check
# ---------------------------------------------------------------------------


class TestUnknownCellHandling:
    """Test that UNKNOWN (-1) cells in OBB are handled correctly.

    Prior bug: _max_cost_in_obb initialized max_cost=0, causing UNKNOWN-only
    OBBs to return 0 (FREE). This led to trajectories heading into unmapped
    territory being marked as collision-free.

    After fix: _max_cost_in_obb initializes max_cost=-1 (UNKNOWN) and only
    updates when observing non-UNKNOWN cells. UNKNOWN-only OBBs return -1,
    which the caller treats as "unknown_penalty" instead of FREE.
    """

    def test_unknown_only_costmap_not_collision_free(self):
        """Trajectory crossing UNKNOWN-only costmap should incur cost, not be free."""
        # Create a costmap that is entirely UNKNOWN (-1)
        unknown_costmap = _make_costmap(
            width=40, height=40, resolution=0.05,
            origin_x=-1.0, origin_y=-1.0,
            fill=CostValues.UNKNOWN,  # all unknown
        )

        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            unknown_penalty=0.8, costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=0.8, sample_step=2,
        )

        # Trajectory stays within the grid (origin at -1.0, robot at 0, heading to 0.5)
        traj = _make_straight_trajectory(dx=0.5, dy=0.0, n_points=20)
        all_traj, all_vals = _make_candidates([traj])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=unknown_costmap,
        )

        # Trajectory should NOT be collision-free; it should incur unknown_penalty
        # With normalized formula: unknown_term = (n_unknown/n_sampled) * cost_threshold * unknown_penalty
        # Fully unknown trajectory: 1.0 * 100 * 0.8 = 80 raw → 64 weighted
        # Minus critic reward (~ 1 * 0.2 = 0.2) → ~63.8
        # This is well above a free trajectory (cost < 0)
        assert result.costs[0] > 0.0, (
            "Trajectory through UNKNOWN territory should have positive cost, "
            "not be collision-free"
        )

    def test_mixed_unknown_and_free_costmap(self):
        """Trajectory crossing UNKNOWN and FREE cells should use max of observed."""
        grid = np.full((40, 40), CostValues.FREE, dtype=np.int8)
        # Leave columns 0-19 as UNKNOWN, columns 20-39 as FREE
        grid[:, 0:20] = CostValues.UNKNOWN

        origin = Pose()
        origin.position.x = -1.0
        origin.position.y = -1.0
        origin.orientation.w = 1.0

        costmap = OccupancyGrid(
            grid=grid, resolution=0.05,
            origin=origin,
            frame_id="map",
        )

        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            unknown_penalty=0.8, costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=0.8, sample_step=2,
        )

        # Trajectory heading into the FREE region (right side)
        traj_into_free = _make_straight_trajectory(dx=0.8, dy=0.0, n_points=20)
        # Trajectory heading into the UNKNOWN region (left side)
        traj_into_unknown = _make_straight_trajectory(dx=-0.8, dy=0.0, n_points=20)

        all_traj, all_vals = _make_candidates([traj_into_free, traj_into_unknown])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_into_free,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )

        # Trajectory into FREE should have lower cost than into UNKNOWN
        assert result.costs[0] < result.costs[1], (
            "Trajectory into FREE region should be cheaper than into UNKNOWN"
        )
        # And trajectory into FREE should be selected
        assert result.index == 0

    def test_unknown_and_obstacle_costmap(self):
        """Trajectory crossing UNKNOWN should be worse than one hitting an obstacle."""
        # Costmap: UNKNOWN on left, FREE in middle, OCCUPIED on right
        grid = np.full((40, 40), CostValues.FREE, dtype=np.int8)
        grid[:, 0:10] = CostValues.UNKNOWN
        grid[:, 30:40] = CostValues.OCCUPIED

        origin = Pose()
        origin.position.x = -1.0
        origin.position.y = -1.0
        origin.orientation.w = 1.0

        costmap = OccupancyGrid(
            grid=grid, resolution=0.05,
            origin=origin,
            frame_id="map",
        )

        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            unknown_penalty=0.5, costmap_weight=0.8, critic_weight=0.2,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=0.8, sample_step=2,
        )

        traj_into_unknown = _make_straight_trajectory(dx=-0.8, dy=0.0, n_points=20)
        traj_into_obstacle = _make_straight_trajectory(dx=0.8, dy=0.0, n_points=20)

        all_traj, all_vals = _make_candidates([traj_into_unknown, traj_into_obstacle])
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_into_unknown,
            all_trajectories=all_traj,
            all_values=all_vals,
            odom=odom,
            traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )

        # Trajectory into UNKNOWN should have some cost, but trajectory into
        # OCCUPIED should be marked as collision and rejected
        assert result.collision_mask[1], "Trajectory into OCCUPIED should be collision"
        assert not result.collision_mask[0], (
            "Trajectory into UNKNOWN should not be marked collision, just expensive"
        )

    def test_unknown_cost_is_bounded_by_horizon_length(self):
        """Unknown cost must be normalized so it doesn't scale with horizon length.

        Bug: previous implementation accumulated unknown_penalty per waypoint,
        causing fully-unknown trajectories with longer horizons to exceed
        max_trajectory_cost regardless of unknown_penalty value.

        With normalized formula: cost = (n_unknown/n_sampled) * cost_threshold *
        unknown_penalty.  A fully-unknown trajectory should produce the same
        unknown cost regardless of how many waypoints fall in the horizon.
        """
        unknown_costmap = _make_costmap(
            width=200, height=200, resolution=0.05,
            origin_x=-5.0, origin_y=-5.0,
            fill=CostValues.UNKNOWN,
        )

        # Two selectors with different horizon lengths but identical params.
        # Both will see fully-unknown trajectories.
        params = dict(
            enabled=True, cost_threshold=50, collision_penalty=1000.0,
            unknown_penalty=0.8, costmap_weight=0.8, critic_weight=0.0,
            robot_length=0.2, robot_half_width=0.1,
            sample_step=2,
        )
        selector_short = TrajectorySelector(horizon_m=0.5, **params)
        selector_long = TrajectorySelector(horizon_m=2.0, **params)

        # Long trajectory so both horizons get many waypoints
        traj = _make_straight_trajectory(dx=3.0, dy=0.0, n_points=60)
        all_traj, all_vals = _make_candidates([traj])
        odom = (0.0, 0.0, 0.0)

        result_short = selector_short.select(
            selected_traj=traj, all_trajectories=all_traj, all_values=all_vals,
            odom=odom, traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=unknown_costmap,
        )
        result_long = selector_long.select(
            selected_traj=traj, all_trajectories=all_traj, all_values=all_vals,
            odom=odom, traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=unknown_costmap,
        )

        # Both should produce the same cost: (1.0 * 50 * 0.8) * 0.8 = 32.0
        # Per the normalization: long horizon does NOT inflate cost.
        expected = 1.0 * 50 * 0.8 * 0.8  # fraction * cost_threshold * unknown_penalty * costmap_weight
        assert abs(result_short.costs[0] - expected) < 0.1, (
            f"Short horizon cost {result_short.costs[0]} != expected {expected}"
        )
        assert abs(result_long.costs[0] - expected) < 0.1, (
            f"Long horizon cost {result_long.costs[0]} != expected {expected}"
        )
        # And critically: both should be BELOW max_trajectory_cost=50
        # so the robot can actually explore unmapped territory
        assert result_long.costs[0] < 50.0, (
            "Fully-unknown trajectory must be below max_trajectory_cost=50 "
            "so the robot can explore unmapped areas"
        )


# ---------------------------------------------------------------------------
# Recency damping (escape dead ends without trail penalty)
# ---------------------------------------------------------------------------


class TestRecencyDamping:
    """Test that recency_damping_count lets the robot retrace recent steps."""

    def test_backward_trajectory_not_penalized_when_damping_enabled(self):
        """A backward trajectory through the most-recent trail should not be penalized."""
        # Trail: robot just walked from x=0 to x=1.0, sampled every 0.25m
        trail = np.array([
            [0.00, 0.0],
            [0.25, 0.0],
            [0.50, 0.0],
            [0.75, 0.0],
            [1.00, 0.0],  # most recent
        ], dtype=np.float32)

        selector = TrajectorySelector(
            enabled=True, explore_weight=10.0, explore_radius=1.0,
            explore_endpoint_bonus=0.0,
            recency_damping_count=3,  # ignore last 3 points
            sample_step=1, horizon_m=2.0,
        )

        # Backward trajectory: robot at (1, 0), heading back through (0.75, 0.5, 0.25)
        # All these points are in the "damped" region of the trail.
        backward_wps = np.array([
            [1.00, 0.0],
            [0.75, 0.0],
            [0.50, 0.0],
            [0.25, 0.0],
        ], dtype=np.float32)

        cost = selector._exploration_cost(backward_wps, trail)
        # With damping=3, only the first 2 trail points (0.0, 0.25) remain.
        # All backward waypoints are within 0.5m of (0.25, 0) → small penalty
        # But should be much less than without damping (where every wp is on trail)

        selector_no_damping = TrajectorySelector(
            enabled=True, explore_weight=10.0, explore_radius=1.0,
            explore_endpoint_bonus=0.0,
            recency_damping_count=0,
            sample_step=1, horizon_m=2.0,
        )
        cost_no_damping = selector_no_damping._exploration_cost(backward_wps, trail)

        # Without damping, robot is on the trail → high penalty
        # With damping, the recent points are dropped → lower penalty
        assert cost < cost_no_damping, (
            f"Recency damping should reduce backward penalty: "
            f"with damping={cost:.3f}, without damping={cost_no_damping:.3f}"
        )

    def test_damping_count_zero_disables_damping(self):
        """recency_damping_count=0 should match old behavior (no points dropped)."""
        trail = np.array([[0.0, 0.0], [0.5, 0.0], [1.0, 0.0]], dtype=np.float32)
        wps = np.array([[1.0, 0.0], [0.75, 0.0], [0.5, 0.0]], dtype=np.float32)

        selector = TrajectorySelector(
            enabled=True, explore_radius=1.0, explore_endpoint_bonus=0.0,
            recency_damping_count=0, sample_step=1, horizon_m=2.0,
        )
        cost = selector._exploration_cost(wps, trail)
        # All waypoints are at distance 0 from a trail point → max penalty
        assert cost > 0.5, f"Expected high penalty without damping, got {cost}"

    def test_damping_larger_than_trail_clears_all(self):
        """If damping > trail size, exploration cost should be zero (no points left)."""
        trail = np.array([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
        wps = np.array([[1.0, 0.0], [0.75, 0.0]], dtype=np.float32)

        selector = TrajectorySelector(
            enabled=True, explore_radius=1.0, explore_endpoint_bonus=0.0,
            recency_damping_count=10,  # > trail size
            sample_step=1, horizon_m=2.0,
        )
        cost = selector._exploration_cost(wps, trail)
        assert cost == 0.0, f"Expected zero cost when damping clears trail, got {cost}"


# ---------------------------------------------------------------------------
# Open-space reward (escape from walls / dead ends)
# ---------------------------------------------------------------------------


class TestOpenSpaceReward:
    """Test that the open-space reward favors trajectories with more FREE cells nearby."""

    def test_open_area_has_higher_reward_than_tight_corridor(self):
        """Trajectory through wide-open area should get higher open-space reward."""
        # Costmap: left half is FREE, right half is OCCUPIED (a wall)
        grid = np.full((40, 40), CostValues.FREE, dtype=np.int8)
        grid[:, 30:] = CostValues.OCCUPIED  # right wall

        origin = Pose()
        origin.position.x = -1.0
        origin.position.y = -1.0
        origin.orientation.w = 1.0
        costmap = OccupancyGrid(
            grid=grid, resolution=0.05, origin=origin, frame_id="map",
        )

        selector = TrajectorySelector(
            enabled=True, open_space_weight=10.0, open_space_radius=0.4,
            sample_step=1, horizon_m=1.0,
        )

        # Waypoints in open area (far from wall)
        wps_open = np.array([
            [-0.5, 0.0], [-0.4, 0.0], [-0.3, 0.0], [-0.2, 0.0],
        ], dtype=np.float32)
        # Waypoints near the wall (right side)
        wps_near_wall = np.array([
            [0.30, 0.0], [0.35, 0.0], [0.40, 0.0], [0.45, 0.0],
        ], dtype=np.float32)

        reward_open = selector._open_space_reward(wps_open, costmap)
        reward_wall = selector._open_space_reward(wps_near_wall, costmap)

        assert reward_open > reward_wall, (
            f"Open area reward {reward_open:.3f} should exceed near-wall {reward_wall:.3f}"
        )
        # Open area should be near 1.0 (all FREE cells around)
        assert reward_open > 0.9, f"Open area reward should be near 1.0, got {reward_open}"

    def test_open_space_reward_zero_for_unknown_costmap(self):
        """All-UNKNOWN costmap should produce zero open-space reward (no FREE cells)."""
        unknown_costmap = _make_costmap(
            width=40, height=40, fill=CostValues.UNKNOWN,
        )
        selector = TrajectorySelector(
            enabled=True, open_space_weight=10.0, open_space_radius=0.4,
            sample_step=1, horizon_m=1.0,
        )
        wps = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32)
        reward = selector._open_space_reward(wps, unknown_costmap)
        assert reward == 0.0, f"UNKNOWN costmap should give zero reward, got {reward}"

    def test_open_space_disabled_when_radius_zero(self):
        """open_space_radius=0 disables the reward."""
        free_costmap = _make_costmap(width=40, height=40, fill=CostValues.FREE)
        selector = TrajectorySelector(
            enabled=True, open_space_weight=10.0, open_space_radius=0.0,
        )
        wps = np.array([[0.0, 0.0], [0.2, 0.0]], dtype=np.float32)
        reward = selector._open_space_reward(wps, free_costmap)
        assert reward == 0.0

    def test_select_prefers_open_trajectory_with_open_space_reward(self):
        """End-to-end: selector should prefer trajectory through open area over near-wall."""
        # Wall on the right side
        grid = np.full((60, 60), CostValues.FREE, dtype=np.int8)
        grid[:, 45:] = CostValues.OCCUPIED

        origin = Pose()
        origin.position.x = -1.5
        origin.position.y = -1.5
        origin.orientation.w = 1.0
        costmap = OccupancyGrid(
            grid=grid, resolution=0.05, origin=origin, frame_id="map",
        )

        selector = TrajectorySelector(
            enabled=True, cost_threshold=100, collision_penalty=1000.0,
            unknown_penalty=0.0, costmap_weight=0.1, critic_weight=0.0,
            open_space_weight=10.0, open_space_radius=0.4,
            robot_length=0.2, robot_half_width=0.1,
            horizon_m=0.6, sample_step=1,
            recency_damping_count=0,
        )

        # Trajectory heading left (away from wall, into open space)
        traj_open = _make_straight_trajectory(dx=-0.5, dy=0.0, n_points=20)
        # Trajectory heading right (toward wall but not into it)
        traj_near_wall = _make_straight_trajectory(dx=0.5, dy=0.0, n_points=20)

        all_traj, all_vals = _make_candidates([traj_near_wall, traj_open])
        # Robot at origin (centre of grid)
        odom = (0.0, 0.0, 0.0)

        result = selector.select(
            selected_traj=traj_open, all_trajectories=all_traj, all_values=all_vals,
            odom=odom, traj_to_waypoints_fn=_identity_waypoint_fn,
            costmap=costmap,
        )

        # Trajectory into open area (index 1) should be selected
        assert result.index == 1, (
            f"Expected open trajectory (idx=1) selected, got idx={result.index}, "
            f"costs={result.costs}"
        )

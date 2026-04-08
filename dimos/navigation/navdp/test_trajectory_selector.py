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

"""Costmap-based trajectory selector for NavDP candidate trajectories.

Evaluates all K candidate trajectories from the diffusion policy against the
occupancy grid costmap (like the A* planner) and/or LiDAR scan points, then
selects the lowest-cost collision-free trajectory.

Collision detection methods:
    1. **Costmap OBB check** — oriented rectangular footprint swept along
       sampled waypoints on the occupancy grid.
    2. **Frenet corridor check** — projects nearby obstacle points into the
       trajectory's curvilinear frame (s, d) and flags any that invade the
       driving corridor (|d| < half_width).
    3. **LiDAR fallback** — point-distance check against scan points when no
       costmap is available.

The selector is a standalone class that can be enabled/disabled at runtime.
When disabled, the original NavDP-selected trajectory is used unchanged.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np

from dimos.msgs.nav_msgs.OccupancyGrid import CostValues, OccupancyGrid

logger = logging.getLogger(__name__)


@dataclass
class SelectionResult:
    """Result of trajectory selection."""

    trajectory: np.ndarray  # (T, 3) selected trajectory in camera frame
    index: int  # index into the candidate array
    cost: float  # total cost of the selected trajectory
    costs: np.ndarray  # (K,) costs for all candidates
    collision_mask: np.ndarray  # (K,) True where trajectory collides
    fallback_used: bool  # True if all trajectories collide


class TrajectorySelector:
    """Selects the best trajectory from NavDP candidates using costmap costs.

    Cost model (per trajectory)::

        costmap_cost     = sum of occupancy grid cell costs along the path
        frenet_cost      = penalty for obstacle points inside the driving corridor
        critic_score     = NavDP critic value (higher = better policy fit)
        exploration_cost = proximity to explored positions (SEEK state only)
        total_cost       = (costmap_cost + frenet_cost) * costmap_weight
                         - critic_score * critic_weight
                         + collision_count * collision_penalty
                         + exploration_cost * explore_weight   (SEEK only)

    The trajectory with the **lowest** total_cost is selected.
    If all trajectories collide, falls back to the original NavDP selection.

    Parameters
    ----------
    enabled : bool
        Master switch. When False, ``select()`` returns the original trajectory.
    cost_threshold : int
        Cells with cost >= this are lethal (impassable). Matches A* convention.
    unknown_penalty : float
        Fraction of cost_threshold applied to unknown (-1) cells.  Applied as
        ``(num_unknown_waypoints / num_sampled_waypoints) * cost_threshold *
        unknown_penalty`` so the contribution is bounded regardless of horizon
        length.  A fully-unknown trajectory contributes
        ``cost_threshold * unknown_penalty`` to the raw costmap cost.
    critic_weight : float
        How much to reward the NavDP critic score (subtracted from cost).
    costmap_weight : float
        Multiplier for accumulated costmap cell costs.
    collision_penalty : float
        Flat penalty added per lethal cell hit along the trajectory.
    robot_length : float
        Full length of the robot front-to-rear (metres).
    robot_half_width : float
        Half the width of the robot side-to-side (metres).
    robot_radius : float
        Circumscribing radius used for LiDAR fallback and obstacle extraction.
    horizon_m : float
        Only evaluate waypoints within this arc-length from the robot.
    sample_step : int
        Evaluate every N-th waypoint (to limit computation).
    max_trajectory_cost : float or None
        If set, the robot halts and waits for the next inference cycle when
        the best non-colliding trajectory's total cost exceeds this value.
        ``None`` (default) disables the check — any non-colliding trajectory
        is executed regardless of cost.
    explore_weight : float
        Multiplier for the exploration cost term (only active during SEEK).
        Higher values push the robot more aggressively into unexplored areas.
    explore_radius : float
        Distance in metres at which the proximity penalty to explored
        positions decays to zero.
    explore_endpoint_bonus : float
        Weight for the endpoint novelty bonus — rewards trajectories whose
        tip points away from explored territory.
    recency_damping_count : int
        Number of most-recent positions in ``explored_positions`` to ignore
        when computing the proximity penalty.  Lets the robot retrace its
        last few steps to escape dead ends without being penalized for
        revisiting cells it just walked over.  At ~0.5 m sampling, a value
        of 3 ignores the last ~1.5 m of trail.
    open_space_weight : float
        Reward weight for the open-space term — subtracted from total cost
        proportional to the fraction of FREE cells around the trajectory.
        Encourages trajectories that head into wide-open areas (away from
        walls and tight corridors).  Set to 0 to disable.
    open_space_radius : float
        Half-width in metres of the box sampled around each waypoint when
        counting nearby FREE cells for the open-space reward.
    depth_obstacle_m : float
        When the minimum depth in the camera's central strip is below this
        distance (metres), forward-pointing trajectories receive a collision
        penalty.  Set to 0 to disable.
    frontier_weight : float
        Reward weight for the frontier direction term.  During SEEK, the
        selector computes the direction toward the nearest frontier
        (free/unknown boundary) on the costmap and rewards trajectories
        whose endpoints align with that direction.  This pulls the robot
        toward unexplored space and helps it escape dead ends.
    """

    def __init__(
        self,
        enabled: bool = True,
        cost_threshold: int = 100,
        unknown_penalty: float = 0.8,
        critic_weight: float = 1.0,
        costmap_weight: float = 0.1,
        collision_penalty: float = 1000.0,
        robot_length: float = 0.6,
        robot_half_width: float = 0.15,
        robot_radius: float = 0.30,
        horizon_m: float = 1.5,
        sample_step: int = 2,
        max_trajectory_cost: float = 50.0,
        min_trajectory_length: float = 0.3,
        critic_min_accept: float = -5.0,
        explore_weight: float = 5.0,
        explore_radius: float = 2.0,
        explore_endpoint_bonus: float = 1.0,
        recency_damping_count: int = 3,
        open_space_weight: float = 5.0,
        open_space_radius: float = 0.6,
        depth_obstacle_m: float = 0.5,
        direction_weight: float = 3.0,
        frontier_weight: float = 5.0,
    ) -> None:
        self._enabled = enabled
        self.cost_threshold = cost_threshold
        self.unknown_penalty = unknown_penalty
        self.critic_weight = critic_weight
        self.costmap_weight = costmap_weight
        self.collision_penalty = collision_penalty
        self.robot_length = robot_length
        self.robot_half_width = robot_half_width
        self.robot_radius = robot_radius
        self.horizon_m = horizon_m
        self.sample_step = max(1, sample_step)
        self.max_trajectory_cost = max_trajectory_cost
        # Degenerate-trajectory filter.  NavDP can emit near-zero-length
        # trajectories that have no collision risk (they don't move) but
        # also don't advance the robot; picking one leads to the robot
        # freezing in front of a wall.  Reject anything whose cumulative
        # path length is below this threshold (metres).
        self.min_trajectory_length = max(0.0, float(min_trajectory_length))
        # NavDP's critic returns ~-10 for trajectories it considers unsafe
        # ("no good direction").  When every surviving candidate has a
        # critic below this value the robot is effectively cornered, even
        # if the footprint check says "no collision" (because the short
        # stay-put trajectory never reaches a wall cell).  Treat this as
        # a stall and ask the caller to escape instead of inching forward.
        self.critic_min_accept = float(critic_min_accept)
        self.explore_weight = explore_weight
        self.explore_radius = max(0.01, explore_radius)
        self.explore_endpoint_bonus = explore_endpoint_bonus
        self.recency_damping_count = max(0, int(recency_damping_count))
        self.open_space_weight = open_space_weight
        self.open_space_radius = max(0.0, open_space_radius)
        self.depth_obstacle_m = depth_obstacle_m
        self.direction_weight = direction_weight
        self.frontier_weight = frontier_weight

        # Cache of the most recent costmap.  The CostMapper typically runs at
        # a lower rate than NavDP (6 Hz inference), so `costmap` is None on
        # many ticks.  Using a stale-but-valid costmap beats disabling the
        # collision check entirely — which is what caused the selector to
        # drive the robot into desks / walls.
        self._latest_costmap: OccupancyGrid | None = None

    # ------------------------------------------------------------------
    # Enable / disable
    # ------------------------------------------------------------------

    @property
    def enabled(self) -> bool:
        return self._enabled

    def enable(self) -> None:
        self._enabled = True
        logger.info("TrajectorySelector enabled")

    def disable(self) -> None:
        self._enabled = False
        logger.info("TrajectorySelector disabled")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def select(
        self,
        selected_traj: np.ndarray,
        all_trajectories: np.ndarray,
        all_values: np.ndarray,
        odom: tuple[float, float, float],
        traj_to_waypoints_fn,
        costmap: OccupancyGrid | None = None,
        scan_points: np.ndarray | None = None,
        explored_positions: np.ndarray | None = None,
        is_seeking: bool = False,
        depth_image: np.ndarray | None = None,
        object_direction: str | None = None,
        frontier_direction: tuple[float, float] | None = None,
    ) -> SelectionResult:
        """Select the best trajectory from candidates.

        Parameters
        ----------
        selected_traj : np.ndarray
            (T, 3) the server-selected trajectory (used as fallback).
        all_trajectories : np.ndarray
            (K, T, 3) all candidate trajectories in camera frame.
        all_values : np.ndarray
            (K,) critic scores for each candidate (higher = better).
        odom : tuple
            (x, y, yaw) current robot pose in world frame.
        traj_to_waypoints_fn : callable
            Converts (T, 3) camera-frame trajectory to (T, 2) base_link waypoints.
            Typically ``TrajectoryController.trajectory_to_waypoints``.
        costmap : OccupancyGrid or None
            Occupancy grid for cost evaluation. Primary collision source.
        scan_points : np.ndarray or None
            (N, 2) LiDAR points in base_link frame. Used as fallback if no costmap.
        explored_positions : np.ndarray or None
            (M, 2) world-frame positions the robot has already visited.
            Used to penalise trajectories heading into explored territory.
            Only applied when ``is_seeking`` is True.
        is_seeking : bool
            True when the state machine is in SEEK (frontier exploration).
            Enables the exploration cost term.
        depth_image : np.ndarray or None
            (H, W) float32 depth image from the robot's camera.  Used for
            forward obstacle detection when the costmap doesn't cover the
            area directly ahead.

        Returns
        -------
        SelectionResult
        """
        # --- Passthrough when disabled or missing data ---
        k = all_trajectories.shape[0] if all_trajectories is not None and all_trajectories.ndim >= 2 else 0
        if not self._enabled or k == 0:
            return SelectionResult(
                trajectory=selected_traj,
                index=-1,
                cost=0.0,
                costs=np.zeros(max(k, 1)),
                collision_mask=np.zeros(max(k, 1), dtype=bool),
                fallback_used=False,
            )

        # --- Costmap cache: CostMapper runs slower than NavDP; reuse the
        # latest known costmap when the caller passes None so obstacle
        # checks stay active on ticks without a fresh map. ---
        if costmap is not None:
            self._latest_costmap = costmap
        else:
            costmap = self._latest_costmap

        # Unpack odom here so the diagnostic probe below can reference ox/oy
        # without causing a NameError (the main loop unpacks it again below).
        ox, oy, oyaw = odom

        # --- Diagnostic: log costmap and trajectory info every ~5s ---
        if not hasattr(self, '_diag_tick'):
            self._diag_tick = 0
        self._diag_tick += 1
        _do_diag = (self._diag_tick % 30 == 1)  # ~every 5s at 6Hz
        if _do_diag:
            if costmap is not None:
                grid = costmap.grid
                n_occupied = int((grid >= self.cost_threshold).sum())
                n_unknown = int((grid == -1).sum())
                n_free = int((grid == 0).sum())
                print(
                    f"[TrajSel-DIAG] costmap {costmap.width}x{costmap.height} "
                    f"res={costmap.resolution:.3f}m "
                    f"origin=({costmap.origin.position.x:.2f},{costmap.origin.position.y:.2f}) "
                    f"cells: free={n_free} occ={n_occupied} unk={n_unknown} "
                    f"threshold={self.cost_threshold}",
                    flush=True,
                )

                # Cost-value distribution — use to pick cost_threshold.
                # Walls should concentrate in the high buckets; floor noise
                # in the low buckets.  Percentiles are computed over observed
                # (non -1) cells only.
                nonneg = grid[grid >= 0]
                if nonneg.size:
                    pct = np.percentile(nonneg, [50, 75, 90, 95, 99]).astype(int).tolist()
                    hist = np.bincount(
                        np.clip(nonneg, 0, 100).astype(np.int64), minlength=101
                    )
                    buckets = [int(hist[i:i + 10].sum()) for i in range(0, 100, 10)]
                    buckets.append(int(hist[100]))  # exact-100 bucket
                    print(
                        f"[TrajSel-DIAG] cost pct50/75/90/95/99={pct} "
                        f"buckets[0-10,10-20,...,90-100,==100]={buckets}",
                        flush=True,
                    )

                # --- Odom ↔ costmap alignment probe ---
                # Sample a 0.5 m box of cells centered on the robot's
                # odom (ox, oy).  If odom is well-aligned with the costmap,
                # max_cost here should be LOW (robot stands on free ground).
                # If odom has drifted and the robot is virtually inside a
                # wall, max_cost will be >= cost_threshold — which explains
                # the apparent "selector ignores walls" behaviour.
                try:
                    res = costmap.resolution
                    half = int(round(0.5 / res))
                    gv = costmap.world_to_grid((ox, oy, 0.0))
                    cx, cy = int(gv.x), int(gv.y)
                    gx0 = max(0, cx - half)
                    gx1 = min(costmap.width, cx + half + 1)
                    gy0 = max(0, cy - half)
                    gy1 = min(costmap.height, cy + half + 1)
                    if gx1 > gx0 and gy1 > gy0:
                        patch = grid[gy0:gy1, gx0:gx1]
                        probe_max = int(patch.max())
                        probe_occ_frac = float(
                            (patch >= self.cost_threshold).sum()
                        ) / patch.size
                        probe_cell_at_robot = (
                            int(grid[cy, cx])
                            if 0 <= cx < costmap.width and 0 <= cy < costmap.height
                            else -999
                        )
                        print(
                            f"[TrajSel-DIAG] odom_probe robot=({ox:.2f},{oy:.2f}) "
                            f"cell_at_robot={probe_cell_at_robot} "
                            f"0.5m_box_max_cost={probe_max} "
                            f"occ_frac={probe_occ_frac:.2%}",
                            flush=True,
                        )
                except Exception as e:  # noqa: BLE001
                    print(f"[TrajSel-DIAG] odom_probe failed: {e}", flush=True)
            else:
                print("[TrajSel-DIAG] costmap=None — collision check disabled", flush=True)

        # Squeeze batch dim: (1, K, T, 3) → (K, T, 3)
        trajs = all_trajectories
        if trajs.ndim == 4:
            trajs = trajs[0]
        vals = all_values
        if vals.ndim == 2:
            vals = vals[0]

        k = trajs.shape[0]
        costs = np.full(k, np.inf)
        collision_mask = np.zeros(k, dtype=bool)

        cos_yaw = math.cos(oyaw)
        sin_yaw = math.sin(oyaw)

        for i in range(k):
            # Convert camera-frame → base_link waypoints
            waypoints_base = traj_to_waypoints_fn(trajs[i])
            if len(waypoints_base) < 2:
                collision_mask[i] = True
                continue

            # Convert base_link → world frame
            waypoints_world = self._base_to_world(
                waypoints_base, ox, oy, cos_yaw, sin_yaw
            )

            # --- Degenerate-trajectory filter ---
            # Short / stationary trajectories pass every collision check
            # trivially (they don't move) but produce tiny (v, w) commands
            # that leave the robot frozen in front of walls.  Treat them
            # as invalid so the selector picks a trajectory that actually
            # advances — or falls back to an escape manoeuvre.
            if self.min_trajectory_length > 0.0 and len(waypoints_base) >= 2:
                seg = np.diff(waypoints_base[:, :2], axis=0)
                path_len = float(np.linalg.norm(seg, axis=1).sum())
                if path_len < self.min_trajectory_length:
                    collision_mask[i] = True
                    costs[i] = self.collision_penalty
                    # Log all degenerate rejections (not just traj#0) so we
                    # can see how many candidates are stay-put on each tick.
                    # print(
                    #     f"[TrajSel] traj#{i} degenerate: path_len={path_len:.3f}m "
                    #     f"< min={self.min_trajectory_length:.2f}m",
                    #     flush=True,
                    # )
                    continue

            # --- Costmap-based collision (OBB footprint) ---
            cost = 0.0
            collides = False
            if costmap is not None:
                cm_cost, cm_collides = self._costmap_cost(
                    waypoints_world, costmap
                )
                cost += cm_cost
                collides = collides or cm_collides

                # --- Frenet corridor check (complementary) ---
                obstacle_pts = self._extract_obstacles_near_trajectory(
                    costmap, waypoints_world, self.robot_radius * 3
                )
                if len(obstacle_pts) > 0:
                    fr_cost, fr_collides = self._frenet_corridor_check(
                        waypoints_world, obstacle_pts, self.robot_half_width
                    )
                    cost += fr_cost
                    collides = collides or fr_collides

                # Diagnostic: show traj #0 waypoint range vs costmap bounds

                if _do_diag and i == 0:
                    wp_min = waypoints_world.min(axis=0)
                    wp_max = waypoints_world.max(axis=0)
                    grid_ox = costmap.origin.position.x
                    grid_oy = costmap.origin.position.y
                    grid_ex = grid_ox + costmap.width * costmap.resolution
                    grid_ey = grid_oy + costmap.height * costmap.resolution
                    in_bounds = (
                        wp_min[0] >= grid_ox and wp_max[0] <= grid_ex and
                        wp_min[1] >= grid_oy and wp_max[1] <= grid_ey
                    )
                    n_obs = len(obstacle_pts)
                    # Sample max cell values along traj #0 waypoints
                    sample_cells = []
                    for wi in range(min(5, len(waypoints_world))):
                        wx_s, wy_s = waypoints_world[wi]
                        gv = costmap.world_to_grid((wx_s, wy_s, 0.0))
                        gx_s, gy_s = int(gv.x), int(gv.y)
                        if 0 <= gx_s < costmap.width and 0 <= gy_s < costmap.height:
                            sample_cells.append(int(costmap.grid[gy_s, gx_s]))
                        else:
                            sample_cells.append("OOB")
                    print(
                        f"[TrajSel-DIAG] traj#0 wp_range x=[{wp_min[0]:.2f},{wp_max[0]:.2f}] "
                        f"y=[{wp_min[1]:.2f},{wp_max[1]:.2f}] "
                        f"costmap_bounds x=[{grid_ox:.2f},{grid_ex:.2f}] y=[{grid_oy:.2f},{grid_ey:.2f}] "
                        f"in_bounds={in_bounds} robot=({ox:.2f},{oy:.2f}) "
                        f"nearby_obs={n_obs} cm_cost={cm_cost:.1f} cm_col={cm_collides} "
                        f"sample_cell_values={sample_cells}",
                        flush=True,
                    )
                    
            elif scan_points is not None:
                # LiDAR fallback
                li_cost, li_collides = self._lidar_cost(
                    waypoints_base, scan_points
                )
                cost += li_cost
                collides = collides or li_collides

            # --- Depth-based forward obstacle detection ---
            # When the centre of the depth image shows a very close obstacle,
            # penalise trajectories that head mostly forward (positive x in
            # base_link).  This catches walls that the costmap misses because
            # its grid doesn't extend far enough ahead of the robot.
            if (
                depth_image is not None
                and self.depth_obstacle_m > 0
                and len(waypoints_base) >= 2
            ):
                depth_cost, depth_collides = self._depth_forward_cost(
                    depth_image, waypoints_base
                )
                cost += depth_cost
                collides = collides or depth_collides

            collision_mask[i] = collides

            # Combined cost: obstacle cost - critic reward + collision penalty
            critic_reward = float(vals[i]) * self.critic_weight if vals is not None else 0.0
            total = cost * self.costmap_weight - critic_reward
            if collides:
                total += self.collision_penalty

            # Exploration cost: penalise trajectories near explored areas (SEEK only)
            if (
                is_seeking
                and explored_positions is not None
                and len(explored_positions) > 0
            ):
                exp_cost = self._exploration_cost(
                    waypoints_world, explored_positions
                )
                total += exp_cost * self.explore_weight

            # Open-space reward: subtract a bonus proportional to the
            # fraction of FREE cells around the trajectory.  Always active
            # (not gated by SEEK) — this is what gives the robot a way to
            # find escape routes from walls and dead ends regardless of state.
            if (
                costmap is not None
                and self.open_space_weight > 0
                and self.open_space_radius > 0
            ):
                open_frac = self._open_space_reward(waypoints_world, costmap)
                total -= open_frac * self.open_space_weight

            # Direction reward: bias toward trajectories heading in the
            # VLM-detected object direction (left/centre/right).
            if (
                object_direction is not None
                and self.direction_weight > 0
                and len(waypoints_base) >= 2
            ):
                dir_reward = self._direction_reward(waypoints_base, object_direction)
                total -= dir_reward * self.direction_weight

            # Frontier direction reward: during SEEK, bias trajectories
            # toward the nearest frontier (free/unknown boundary) on the
            # costmap.  This gives the robot a persistent pull toward
            # unexplored space and helps it escape dead ends and rooms.
            if (
                is_seeking
                and frontier_direction is not None
                and self.frontier_weight > 0
                and len(waypoints_world) >= 2
            ):
                fr_reward = self._frontier_direction_reward(
                    waypoints_world, frontier_direction
                )
                total -= fr_reward * self.frontier_weight

            costs[i] = total

        # Select best non-colliding trajectory
        free_indices = np.where(~collision_mask)[0]
        n_colliding = int(collision_mask.sum())
        n_free = k - n_colliding
        source = "costmap+frenet" if costmap is not None else ("lidar" if scan_points is not None else "none")

        if len(free_indices) > 0:
            best_idx = int(free_indices[np.argmin(costs[free_indices])])
            best_cost = float(costs[best_idx])
            best_critic = float(vals[best_idx]) if vals is not None else 0.0
            print(
                f"[TrajectorySelector] {n_free}/{k} candidates collision-free "
                f"(source={source}) → selected traj #{best_idx} "
                f"(cost={best_cost:.1f}, critic={best_critic:.3f})",
                flush=True,
            )
            # Per-component cost breakdown for the winner (every diag tick).
            if _do_diag and vals is not None:
                _w = best_idx
                _cm_cost_w, _ = self._costmap_cost(
                    self._base_to_world(
                        traj_to_waypoints_fn(trajs[_w]),
                        ox, oy, cos_yaw, sin_yaw,
                    ),
                    costmap,
                ) if costmap is not None else (0.0, False)
                _critic_r = float(vals[_w]) * self.critic_weight
                _open_f = (
                    self._open_space_reward(
                        self._base_to_world(
                            traj_to_waypoints_fn(trajs[_w]),
                            ox, oy, cos_yaw, sin_yaw,
                        ),
                        costmap,
                    )
                    if costmap is not None else 0.0
                )
                print(
                    f"[TrajSel-DIAG] winner traj#{_w} breakdown: "
                    f"cm_cost={_cm_cost_w * self.costmap_weight:.2f} "
                    f"critic_reward={_critic_r:.2f} "
                    f"open_space_reward={_open_f * self.open_space_weight:.2f} "
                    f"total={best_cost:.2f}",
                    flush=True,
                )
            if (
                self.max_trajectory_cost is not None
                and best_cost > self.max_trajectory_cost
            ):
                print(
                    f"[TrajectorySelector] Best cost {best_cost:.1f} exceeds "
                    f"threshold {self.max_trajectory_cost:.1f} → rejecting, robot should wait",
                    flush=True,
                )
                return SelectionResult(
                    trajectory=selected_traj,
                    index=-1,
                    cost=best_cost,
                    costs=costs,
                    collision_mask=collision_mask,
                    fallback_used=True,
                )

            # Critic floor: NavDP itself says "this trajectory is unsafe"
            # (critic ≈ -10) for every survivor.  Don't pick one — signal
            # stall so the caller can trigger an escape/reorient.
            if (
                vals is not None
                and best_critic < self.critic_min_accept
            ):
                print(
                    f"[TrajectorySelector] Best critic {best_critic:.2f} below "
                    f"floor {self.critic_min_accept:.2f} → all candidates unsafe, "
                    f"rejecting to trigger escape",
                    flush=True,
                )
                return SelectionResult(
                    trajectory=selected_traj,
                    index=-1,
                    cost=best_cost,
                    costs=costs,
                    collision_mask=collision_mask,
                    fallback_used=True,
                )
            return SelectionResult(
                trajectory=trajs[best_idx],
                index=best_idx,
                cost=best_cost,
                costs=costs,
                collision_mask=collision_mask,
                fallback_used=False,
            )

        # All collide — signal caller to halt and wait for next inference
        print(
            f"[TrajectorySelector] ALL {k}/{k} candidates collide "
            f"(source={source}) → rejecting all, robot should wait",
            flush=True,
        )
        return SelectionResult(
            trajectory=selected_traj,
            index=-1,
            cost=float(costs.min()),
            costs=costs,
            collision_mask=collision_mask,
            fallback_used=True,
        )

    # ------------------------------------------------------------------
    # Internal cost functions
    # ------------------------------------------------------------------

    def _base_to_world(
        self,
        waypoints_base: np.ndarray,
        ox: float,
        oy: float,
        cos_yaw: float,
        sin_yaw: float,
    ) -> np.ndarray:
        """Transform (T, 2) base_link waypoints to world frame."""
        wx = ox + cos_yaw * waypoints_base[:, 0] - sin_yaw * waypoints_base[:, 1]
        wy = oy + sin_yaw * waypoints_base[:, 0] + cos_yaw * waypoints_base[:, 1]
        return np.stack([wx, wy], axis=1)

    def _costmap_cost(
        self, waypoints_world: np.ndarray, costmap: OccupancyGrid
    ) -> tuple[float, bool]:
        """Evaluate trajectory cost using an oriented rectangular footprint.

        For each sampled waypoint (within horizon), compute the local heading
        from consecutive waypoints, then check an oriented bounding box (OBB)
        of size ``robot_length x 2*robot_half_width`` against the occupancy grid.

        Returns (accumulated_cost, has_collision).
        """
        indices = self._horizon_indices(waypoints_world)
        if len(indices) == 0:
            return 0.0, False

        # Track observed-cell cost separately from unknown-waypoint count.
        # The UNKNOWN contribution is normalized to the *fraction* of unknown
        # waypoints so a long exploration trajectory through unmapped territory
        # is not penalized proportional to its length.  Without normalization,
        # any trajectory entering a few unmapped cells accumulates enough cost
        # to exceed max_trajectory_cost and the robot cannot explore.
        observed_cost = 0.0
        n_sampled = 0
        n_unknown = 0
        has_collision = False
        half_len = self.robot_length / 2.0

        # Pre-compute grid bounds for the out-of-bounds check
        _grid_ox = costmap.origin.position.x
        _grid_oy = costmap.origin.position.y
        _grid_max_x = _grid_ox + costmap.width * costmap.resolution
        _grid_max_y = _grid_oy + costmap.height * costmap.resolution

        for idx in indices:
            wx, wy = waypoints_world[idx]
            n_sampled += 1

            # If waypoint centre is outside the costmap grid, the robot
            # would be driving into unmapped territory.  Treat as collision
            # so the selector never picks a trajectory heading off the map.
            if not (_grid_ox <= wx <= _grid_max_x and _grid_oy <= wy <= _grid_max_y):
                has_collision = True
                observed_cost += self.collision_penalty
                continue

            # Compute local heading from trajectory direction
            if idx + 1 < len(waypoints_world):
                dx = waypoints_world[idx + 1, 0] - wx
                dy = waypoints_world[idx + 1, 1] - wy
            elif idx > 0:
                dx = wx - waypoints_world[idx - 1, 0]
                dy = wy - waypoints_world[idx - 1, 1]
            else:
                dx, dy = 1.0, 0.0
            heading = math.atan2(dy, dx)

            cell_max = self._max_cost_in_obb(
                costmap, wx, wy, heading, half_len, self.robot_half_width
            )

            if cell_max >= self.cost_threshold:
                has_collision = True
                observed_cost += self.collision_penalty
                if not hasattr(self, '_obb_collision_logged'):
                    self._obb_collision_logged = True
                    print(
                        f"[TrajSel] OBB collision: cell_max={cell_max} >= threshold={self.cost_threshold} "
                        f"at world=({wx:.2f},{wy:.2f})",
                        flush=True,
                    )
            elif cell_max == CostValues.UNKNOWN:
                n_unknown += 1
            else:
                observed_cost += max(0, cell_max)

        # Normalized unknown term: fraction of unknown waypoints scaled by
        # cost_threshold * unknown_penalty.  Bounded by [0, cost_threshold * unknown_penalty].
        if n_sampled > 0:
            unknown_fraction = n_unknown / n_sampled
            unknown_term = unknown_fraction * self.cost_threshold * self.unknown_penalty
        else:
            unknown_term = 0.0

        return observed_cost + unknown_term, has_collision

    def _frenet_corridor_check(
        self,
        waypoints_world: np.ndarray,
        obstacle_points: np.ndarray,
        corridor_half_width: float,
    ) -> tuple[float, bool]:
        """Check obstacles in the trajectory's Frenet (curvilinear) frame.

        For each obstacle point, find the closest trajectory segment and
        project onto tangent (s) and normal (d). If ``|d| < corridor_half_width``
        the obstacle invades the driving corridor.

        Returns (accumulated_cost, has_collision).
        """
        indices = self._horizon_indices(waypoints_world)
        if len(indices) < 2 or len(obstacle_points) == 0:
            return 0.0, False

        # Build segments from sampled waypoints
        wp = waypoints_world[indices]
        seg_starts = wp[:-1]  # (S, 2)
        seg_ends = wp[1:]     # (S, 2)
        seg_vecs = seg_ends - seg_starts  # (S, 2)
        seg_lens = np.linalg.norm(seg_vecs, axis=1)  # (S,)

        # Skip degenerate segments
        valid = seg_lens > 1e-6
        if not np.any(valid):
            return 0.0, False
        seg_starts = seg_starts[valid]
        seg_ends = seg_ends[valid]
        seg_vecs = seg_vecs[valid]
        seg_lens = seg_lens[valid]

        # Unit tangent and normal for each segment
        tangents = seg_vecs / seg_lens[:, None]  # (S, 2)
        # Normal: rotate tangent 90° counter-clockwise
        normals = np.stack([-tangents[:, 1], tangents[:, 0]], axis=1)  # (S, 2)

        n_obs = len(obstacle_points)

        total_cost = 0.0
        has_collision = False

        # Vectorized: for each obstacle, find closest segment and project
        # Expand: obs (N, 2), seg_starts (S, 2) → diff (N, S, 2)
        diff = obstacle_points[:, None, :] - seg_starts[None, :, :]  # (N, S, 2)

        # Project onto tangent: s = dot(diff, tangent)
        s_proj = np.sum(diff * tangents[None, :, :], axis=2)  # (N, S)
        # Clamp s to [0, seg_len]
        s_clamped = np.clip(s_proj, 0.0, seg_lens[None, :])  # (N, S)

        # Closest point on segment: start + s_clamped * tangent
        closest = seg_starts[None, :, :] + s_clamped[:, :, None] * tangents[None, :, :]  # (N, S, 2)
        dist_vec = obstacle_points[:, None, :] - closest  # (N, S, 2)
        dist_sq = np.sum(dist_vec ** 2, axis=2)  # (N, S)

        # Find nearest segment for each obstacle
        nearest_seg = np.argmin(dist_sq, axis=1)  # (N,)
        obs_idx = np.arange(n_obs)

        # Lateral distance (d) = dot(diff_to_nearest, normal_of_nearest)
        d_vals = np.sum(
            diff[obs_idx, nearest_seg, :] * normals[nearest_seg, :], axis=1
        )  # (N,)
        s_vals = s_proj[obs_idx, nearest_seg]  # (N,)
        s_max = seg_lens[nearest_seg]

        # Obstacle is in corridor if |d| < half_width AND 0 <= s <= seg_len
        in_corridor = (
            (np.abs(d_vals) < corridor_half_width)
            & (s_vals >= -0.05)  # small tolerance for points near segment start
            & (s_vals <= s_max + 0.05)
        )

        n_invading = int(np.sum(in_corridor))
        if n_invading > 0:
            has_collision = True
            total_cost += float(n_invading) * self.collision_penalty

        return total_cost, has_collision

    def _lidar_cost(
        self, waypoints_base: np.ndarray, scan_points: np.ndarray
    ) -> tuple[float, bool]:
        """Evaluate trajectory cost using LiDAR scan points (fallback).

        Checks whether any scan point falls within robot_radius of a waypoint.
        Cost is proportional to the number of nearby obstacles.
        """
        indices = self._horizon_indices(waypoints_base)
        if len(indices) == 0 or len(scan_points) == 0:
            return 0.0, False

        # Only forward scan points
        fwd = scan_points[scan_points[:, 0] >= -0.1]
        if len(fwd) == 0:
            return 0.0, False

        r_sq = self.robot_radius ** 2
        total_cost = 0.0
        has_collision = False

        for idx in indices:
            dx = fwd[:, 0] - waypoints_base[idx, 0]
            dy = fwd[:, 1] - waypoints_base[idx, 1]
            dist_sq = dx * dx + dy * dy
            nearby = np.sum(dist_sq < r_sq)
            if nearby > 0:
                has_collision = True
                total_cost += float(nearby) * self.collision_penalty
            else:
                # Proximity cost: inverse distance to nearest obstacle
                min_dist = float(np.sqrt(dist_sq.min()))
                if min_dist < self.robot_radius * 3:
                    total_cost += (self.robot_radius * 3 - min_dist) * 50.0

        return total_cost, has_collision

    def _exploration_cost(
        self,
        waypoints_world: np.ndarray,
        explored_positions: np.ndarray,
    ) -> float:
        """Penalise trajectories that pass through already-explored areas.

        For each sampled waypoint within the horizon, compute the minimum
        distance to any explored position.  Waypoints close to explored
        positions incur a proximity penalty (linear decay to zero at
        ``explore_radius``).  The trajectory endpoint receives a novelty
        bonus when it points away from explored territory.

        Returns a scalar cost in ``[-explore_endpoint_bonus, 1.0]``.
        Positive means "heading into explored area", negative means
        "heading into novel territory".
        """
        indices = self._horizon_indices(waypoints_world)
        if len(indices) == 0 or len(explored_positions) == 0:
            return 0.0

        wps = waypoints_world[indices]  # (N, 2)
        # Recency damping: drop the most-recent positions of the trail so the
        # robot can retrace its last few steps without penalty.  This is
        # critical for escaping dead ends — without it, every backward
        # trajectory passes through trail points and is penalized.
        if self.recency_damping_count > 0:
            exp = explored_positions[: -self.recency_damping_count]
        else:
            exp = explored_positions
        if len(exp) == 0:
            return 0.0  # no points left after damping

        # Vectorised pairwise distances: (N, M)
        # Using broadcasting instead of scipy to avoid an extra dependency.
        diff = wps[:, None, :] - exp[None, :, :]  # (N, M, 2)
        dists = np.sqrt(np.sum(diff * diff, axis=2))  # (N, M)
        min_dists = dists.min(axis=1)  # (N,)

        # Per-waypoint proximity penalty: 1.0 at dist=0, 0.0 at dist>=radius
        penalties = np.clip(1.0 - min_dists / self.explore_radius, 0.0, 1.0)
        avg_penalty = float(penalties.mean())

        # Endpoint novelty bonus: reward heading into unexplored territory
        endpoint_min = float(min_dists[-1])
        novelty_bonus = (
            min(endpoint_min / self.explore_radius, 1.0)
            * self.explore_endpoint_bonus
        )

        return avg_penalty - novelty_bonus

    def _open_space_reward(
        self,
        waypoints_world: np.ndarray,
        costmap: OccupancyGrid,
    ) -> float:
        """Compute the average fraction of FREE cells around the trajectory.

        For each sampled waypoint within the horizon, count cells inside a
        square of half-width ``open_space_radius`` (in metres) around the
        waypoint and report the fraction that are FREE (cost == 0).
        UNKNOWN and OCCUPIED cells are excluded from the numerator.

        Returns a value in ``[0, 1]``.  Higher means the trajectory passes
        through more open space — used as a reward (subtracted from total
        cost) so the selector prefers wider corridors over tight ones and
        gives the robot a way to ESCAPE walls/dead-ends by heading toward
        the most-open direction.
        """
        if self.open_space_radius <= 0:
            return 0.0
        indices = self._horizon_indices(waypoints_world)
        if len(indices) == 0:
            return 0.0

        res = costmap.resolution
        if res <= 0:
            return 0.0
        radius_cells = max(1, int(math.ceil(self.open_space_radius / res)))
        grid = costmap.grid
        h, w = grid.shape

        free_fractions = []
        for idx in indices:
            wx, wy = waypoints_world[idx]
            gv = costmap.world_to_grid((wx, wy, 0.0))
            gx, gy = int(gv.x), int(gv.y)

            # Box bounds clipped to grid
            x_min = max(0, gx - radius_cells)
            x_max = min(w - 1, gx + radius_cells)
            y_min = max(0, gy - radius_cells)
            y_max = min(h - 1, gy + radius_cells)
            if x_min > x_max or y_min > y_max:
                continue

            patch = grid[y_min:y_max + 1, x_min:x_max + 1]
            n_total = patch.size
            if n_total == 0:
                continue
            n_free = int(np.count_nonzero(patch == CostValues.FREE))
            free_fractions.append(n_free / n_total)

        if not free_fractions:
            return 0.0
        return float(np.mean(free_fractions))

    def _depth_forward_cost(
        self,
        depth_image: np.ndarray,
        waypoints_base: np.ndarray,
    ) -> tuple[float, bool]:
        """Penalise forward-heading trajectories when depth shows an obstacle ahead.

        Two-tier penalty:
        - Hard collision (depth < depth_obstacle_m, i.e. 0.5m): full collision
          flag + full penalty.  Catches walls that are in the robot's immediate
          path but not yet in the costmap.
        - Soft warning (depth < 2*depth_obstacle_m, i.e. 1.0m): partial penalty
          proportional to proximity.  Discourages heading toward walls 0.5-1.0m
          away before they trigger the hard threshold.

        Samples the central vertical strip of the depth image (middle 40% of
        width, middle 60% of height) and computes the minimum valid depth.

        Returns (cost, has_collision).
        """
        h, w = depth_image.shape[:2]

        # Central strip of the image (where the robot is heading)
        x0, x1 = int(w * 0.3), int(w * 0.7)
        y0, y1 = int(h * 0.2), int(h * 0.8)
        centre = depth_image[y0:y1, x0:x1]

        # Filter invalid depth (zero or NaN)
        valid = centre[(centre > 0.05) & np.isfinite(centre)]
        if len(valid) == 0:
            return 0.0, False

        min_depth = float(valid.min())
        soft_threshold = self.depth_obstacle_m * 2.0

        if min_depth >= soft_threshold:
            return 0.0, False

        # Only penalise trajectories heading forward (positive x in base_link)
        endpoint = waypoints_base[-1]
        forward_component = endpoint[0]  # x in base_link = forward
        if forward_component <= 0:
            # Trajectory turns backward / sideways — not heading into obstacle
            return 0.0, False

        forward_fraction = min(forward_component / self.horizon_m, 1.0)

        if min_depth < self.depth_obstacle_m:
            # Hard tier: depth below hard threshold — full collision
            severity = 1.0 - min_depth / self.depth_obstacle_m
            cost = severity * forward_fraction * self.collision_penalty
            has_collision = severity > 0.5 and forward_fraction > 0.2
        else:
            # Soft tier: between depth_obstacle_m and 2*depth_obstacle_m
            # Linear penalty, no hard collision flag
            severity = 1.0 - (min_depth - self.depth_obstacle_m) / self.depth_obstacle_m
            cost = severity * forward_fraction * self.collision_penalty * 0.4
            has_collision = False

        return cost, has_collision

    def _direction_reward(
        self,
        waypoints_base: np.ndarray,
        direction: str,
    ) -> float:
        """Reward trajectories heading in the indicated direction.

        The VLN skill classifies the object as left/centre/right in the
        camera image and passes it here.  Returns a value in [0, 1] where
        1 means the trajectory endpoint perfectly aligns with the requested
        direction.

        In base_link frame: +x = forward, +y = left.
        """
        endpoint = waypoints_base[-1]  # (x, y) in base_link
        dist = math.sqrt(endpoint[0] ** 2 + endpoint[1] ** 2)
        if dist < 1e-6:
            return 0.5
        nx, ny = endpoint[0] / dist, endpoint[1] / dist
        if direction == "left":
            return max(0.0, ny)   # +y = left in base_link
        elif direction == "right":
            return max(0.0, -ny)  # -y = right in base_link
        else:  # centre
            return max(0.0, nx)   # +x = forward in base_link

    def _frontier_direction_reward(
        self,
        waypoints_world: np.ndarray,
        frontier_direction: tuple[float, float],
    ) -> float:
        """Reward trajectories whose endpoint heads toward the frontier direction.

        ``frontier_direction`` is a unit vector in world frame pointing from
        the robot toward the nearest frontier centroid.  The reward is the
        cosine similarity between that vector and the trajectory's
        displacement vector (start → endpoint in world frame), mapped to
        ``[0, 1]``.

        Returns 1.0 when perfectly aligned, 0.0 when orthogonal or opposing.
        """
        indices = self._horizon_indices(waypoints_world)
        if len(indices) < 2:
            return 0.5  # neutral

        start = waypoints_world[indices[0]]
        end = waypoints_world[indices[-1]]
        traj_vec = end - start
        traj_len = math.sqrt(traj_vec[0] ** 2 + traj_vec[1] ** 2)
        if traj_len < 1e-6:
            return 0.5

        # Normalise
        tx, ty = traj_vec[0] / traj_len, traj_vec[1] / traj_len
        fx, fy = frontier_direction
        # Cosine similarity → [-1, 1] → map to [0, 1]
        cos_sim = tx * fx + ty * fy
        return max(0.0, (cos_sim + 1.0) / 2.0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _horizon_indices(self, waypoints: np.ndarray) -> list[int]:
        """Return sampled waypoint indices within the arc-length horizon."""
        if len(waypoints) < 2:
            return [0] if len(waypoints) == 1 else []

        diffs = np.diff(waypoints, axis=0)
        seg_lengths = np.linalg.norm(diffs, axis=1)
        cum_dist = np.concatenate([[0.0], np.cumsum(seg_lengths)])

        last_idx = int(np.searchsorted(cum_dist, self.horizon_m, side="right")) - 1
        last_idx = max(0, min(last_idx, len(waypoints) - 1))

        indices = list(range(0, last_idx + 1, self.sample_step))
        if indices and indices[-1] != last_idx:
            indices.append(last_idx)
        return indices

    def _max_cost_in_obb(
        self,
        costmap: OccupancyGrid,
        wx: float,
        wy: float,
        heading: float,
        half_length: float,
        half_width: float,
    ) -> int:
        """Return the max cell cost within an oriented bounding box on the grid.

        The OBB is centred at (wx, wy) in world coordinates, oriented along
        ``heading``, with extents ``half_length`` (along heading) and
        ``half_width`` (perpendicular).

        We iterate over the axis-aligned bounding box of the OBB in grid space
        and test each cell centre against the rotated rectangle.

        Returns the maximum cost of observed (non-UNKNOWN) cells, or UNKNOWN
        if no observed cells are within the OBB.
        """
        cos_h = math.cos(heading)
        sin_h = math.sin(heading)

        # Four corners of the OBB in world frame
        # local frame: x = along heading, y = perpendicular
        corners_local = [
            (+half_length, +half_width),
            (+half_length, -half_width),
            (-half_length, +half_width),
            (-half_length, -half_width),
        ]
        corners_world = []
        for lx, ly in corners_local:
            cx = wx + cos_h * lx - sin_h * ly
            cy = wy + sin_h * lx + cos_h * ly
            corners_world.append((cx, cy))

        # Convert corners to grid coordinates and find AABB
        gxs = []
        gys = []
        for cx, cy in corners_world:
            gv = costmap.world_to_grid((cx, cy, 0.0))
            gxs.append(gv.x)
            gys.append(gv.y)

        gx_min = max(0, int(math.floor(min(gxs))))
        gx_max = min(costmap.width - 1, int(math.ceil(max(gxs))))
        gy_min = max(0, int(math.floor(min(gys))))
        gy_max = min(costmap.height - 1, int(math.ceil(max(gys))))

        if gx_min > gx_max or gy_min > gy_max:
            return int(CostValues.UNKNOWN)

        # Check each cell in the AABB; keep only those inside the OBB
        # Track whether we've found any observed cells (not UNKNOWN).
        # If all cells in the OBB are UNKNOWN, return UNKNOWN so the caller
        # can apply unknown_penalty rather than treating as FREE.
        max_cost = int(CostValues.UNKNOWN)  # -1
        res = costmap.resolution
        ox = costmap.origin.position.x
        oy = costmap.origin.position.y

        for gy in range(gy_min, gy_max + 1):
            for gx in range(gx_min, gx_max + 1):
                # Grid cell centre → world
                cell_wx = ox + (gx + 0.5) * res
                cell_wy = oy + (gy + 0.5) * res
                # Transform to OBB local frame
                dx = cell_wx - wx
                dy = cell_wy - wy
                local_x = cos_h * dx + sin_h * dy
                local_y = -sin_h * dx + cos_h * dy
                # Inside OBB?
                if abs(local_x) <= half_length and abs(local_y) <= half_width:
                    cell_cost = int(costmap.grid[gy, gx])
                    # Skip UNKNOWN cells; they don't contribute to max_cost
                    # but if we see them we know the OBB is not entirely unmapped.
                    if cell_cost != CostValues.UNKNOWN:
                        if max_cost == CostValues.UNKNOWN:
                            # First observed cell
                            max_cost = cell_cost
                        elif cell_cost > max_cost:
                            # Update max
                            max_cost = cell_cost

        return max_cost

    def _extract_obstacles_near_trajectory(
        self,
        costmap: OccupancyGrid,
        waypoints_world: np.ndarray,
        radius: float,
    ) -> np.ndarray:
        """Extract occupied cell centres near the trajectory as world-frame points.

        Returns (N, 2) array of obstacle positions within ``radius`` of any
        waypoint, used as input to the Frenet corridor check.
        """
        if len(waypoints_world) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Trajectory bounding box in world frame, expanded by radius
        traj_min = waypoints_world.min(axis=0) - radius
        traj_max = waypoints_world.max(axis=0) + radius

        # Convert to grid bounds
        gv_min = costmap.world_to_grid((traj_min[0], traj_min[1], 0.0))
        gv_max = costmap.world_to_grid((traj_max[0], traj_max[1], 0.0))
        gx_min = max(0, int(math.floor(gv_min.x)))
        gx_max = min(costmap.width - 1, int(math.ceil(gv_max.x)))
        gy_min = max(0, int(math.floor(min(gv_min.y, gv_max.y))))
        gy_max = min(costmap.height - 1, int(math.ceil(max(gv_min.y, gv_max.y))))

        if gx_min > gx_max or gy_min > gy_max:
            return np.zeros((0, 2), dtype=np.float32)

        # Extract occupied cells in the region
        patch = costmap.grid[gy_min:gy_max + 1, gx_min:gx_max + 1]
        # Use the selector's configured cost_threshold (typ. 50) rather than
        # CostValues.OCCUPIED (=100).  Gradient-based costmaps
        # (height_cost_occupancy) emit graded values in [0,100]; most wall
        # cells land in 50–99, so requiring ==100 silently drops them and
        # Frenet corridor checks never see the walls.
        occ_ys, occ_xs = np.where(patch >= self.cost_threshold)

        if len(occ_ys) == 0:
            return np.zeros((0, 2), dtype=np.float32)

        # Convert grid indices back to world coordinates
        res = costmap.resolution
        ox = costmap.origin.position.x
        oy = costmap.origin.position.y
        world_xs = ox + (occ_xs + gx_min + 0.5) * res
        world_ys = oy + (occ_ys + gy_min + 0.5) * res

        return np.stack([world_xs, world_ys], axis=1).astype(np.float32)

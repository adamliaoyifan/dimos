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

"""Typed sample schema — the single contract between adapters and bridges.

All types are pure Python dataclasses.  No ROS imports, no DimOS imports.
Adapters produce these; bridges consume them.

Naming convention:
  - *Sample  — raw sensor/state datum (one message per publish)
  - *Record  — stored/indexed semantic entity
  - *Event   — discrete notification (e.g. goal reached, memory hit)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


@dataclass
class Pose2D:
    """2-D pose in a named frame."""

    x: float
    y: float
    yaw: float  # radians
    stamp_ns: int  # nanoseconds since epoch (use ClockPublisher.now_ns())
    frame: str = "odom"


@dataclass
class Pose3D:
    """3-D pose (position + quaternion) in a named frame."""

    x: float
    y: float
    z: float
    qx: float
    qy: float
    qz: float
    qw: float
    stamp_ns: int
    frame: str = "odom"


# ---------------------------------------------------------------------------
# Sensor samples
# ---------------------------------------------------------------------------


@dataclass
class OdomSample:
    """Robot odometry: pose + velocity."""

    pose: Pose3D
    vx: float = 0.0  # m/s forward
    vy: float = 0.0
    vz: float = 0.0
    wx: float = 0.0  # rad/s
    wy: float = 0.0
    wz: float = 0.0
    child_frame: str = "base_link"


@dataclass
class ImageSample:
    """Raw or JPEG-compressed image."""

    data: bytes  # raw RGB24 or JPEG bytes
    width: int
    height: int
    encoding: str  # "rgb8", "bgr8", "jpeg", "mono8"
    stamp_ns: int
    frame: str = "camera"


@dataclass
class PointCloudSample:
    """Unorganized point cloud (x, y, z float32, row-major)."""

    points_xyz: bytes  # raw float32 triples, little-endian
    num_points: int
    stamp_ns: int
    frame: str = "lidar"
    intensity: bytes | None = None  # float32 per point, same length


@dataclass
class CostmapSample:
    """2-D occupancy grid sample."""

    data: bytes  # int8 row-major (height × width); -1=unknown, 0=free, 1-100=occ
    width: int
    height: int
    resolution: float  # m/cell
    origin_x: float  # world X of grid cell (0, 0)
    origin_y: float
    stamp_ns: int
    frame: str = "map"


# ---------------------------------------------------------------------------
# Navigation samples
# ---------------------------------------------------------------------------


@dataclass
class WaypointMeta:
    """Per-waypoint metadata attached to a trajectory."""

    critic_score: float = 0.0
    cost: float = 0.0
    speed: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajSample:
    """NavDP (or other planner) trajectory candidate."""

    traj_id: str
    points: list[Pose2D]
    waypoint_meta: list[WaypointMeta]  # same length as points
    is_selected: bool = False
    color_rgb: tuple[float, float, float] = (0.2, 0.8, 1.0)
    stamp_ns: int = 0
    frame: str = "odom"
    source: str = ""  # e.g. "navdp", "astar"


@dataclass
class PathSample:
    """Executed path history (grow-only trail of poses)."""

    poses: list[Pose2D]
    stamp_ns: int
    frame: str = "odom"


@dataclass
class FrontierSample:
    """Frontier exploration candidates with scores."""

    @dataclass
    class Frontier:
        x: float
        y: float
        score: float
        info_gain: float = 0.0
        memory_novelty: float = 0.0
        corridor_score: float = 0.0
        vlm_confidence: float | None = None
        rank: int = 0

    frontiers: list[Frontier]
    stamp_ns: int
    frame: str = "map"


# ---------------------------------------------------------------------------
# Semantic memory
# ---------------------------------------------------------------------------


@dataclass
class MemoryRecord:
    """One spatial-memory entry (image + pose + tags)."""

    record_id: str  # unique, stable across updates
    pose: Pose2D
    thumbnail: bytes | None  # JPEG bytes, may be None for non-image entries
    tags: list[str] = field(default_factory=list)
    description: str = ""
    similarity: float | None = None  # set only on query-result hits
    embedding_hash: str | None = None


@dataclass
class MemoryQueryEvent:
    """Fired when SpatialMemory.query_by_text / query_by_location returns results."""

    query_text: str
    results: list[MemoryRecord]
    stamp_ns: int


# ---------------------------------------------------------------------------
# Robot geometry (for interactive marker)
# ---------------------------------------------------------------------------


@dataclass
class RobotGeometry:
    """Static robot body description published once at startup."""

    name: str
    length: float  # m, fore-aft
    width: float  # m, lateral
    height: float  # m
    base_link_frame: str = "base_link"
    mesh_resource: str | None = None  # package://... URI for URDF mesh
    mass_kg: float | None = None
    sensor_mounts: list[dict[str, Any]] = field(default_factory=list)
    # e.g. [{"name": "lidar", "frame": "lidar", "offset_xyz": [0.1, 0, 0.3]}]


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dataclass
class NavigationEvent:
    """Discrete navigation state change."""

    state: str  # "IDLE" | "SEEK" | "APPROACH" | "STOPPED" | "FAILED"
    goal_x: float | None = None
    goal_y: float | None = None
    message: str = ""
    stamp_ns: int = 0

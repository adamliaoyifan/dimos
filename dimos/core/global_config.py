# Copyright 2025-2026 Dimensional Inc.
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

import re
from typing import Literal, TypeAlias

from pydantic_settings import BaseSettings, SettingsConfigDict

from dimos.models.vl.types import VlModelName

ViewerBackend: TypeAlias = Literal["rerun", "rerun-web", "rerun-connect", "foxglove", "none"]


def _get_all_numbers(s: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]


class GlobalConfig(BaseSettings):
    robot_ip: str | None = None
    robot_ips: str | None = None
    xarm7_ip: str | None = None
    xarm6_ip: str | None = None
    can_port: str = "can0"
    simulation: bool = False
    replay: bool = False
    replay_dir: str = "go2_sf_office"
    new_memory: bool = False
    viewer: ViewerBackend = "rerun"
    n_workers: int = 2
    memory_limit: str = "auto"
    mujoco_camera_position: str | None = None
    mujoco_room: str | None = None
    mujoco_room_from_occupancy: str | None = None
    mujoco_global_costmap_from_occupancy: str | None = None
    mujoco_global_map_from_pointcloud: str | None = None
    mujoco_start_pos: str = "-1.0, 1.0"
    mujoco_steps_per_frame: int = 7
    robot_model: str | None = None
    robot_width: float = 0.3
    robot_rotation_diameter: float = 0.6
    nerf_speed: float = 1.0
    planner_robot_speed: float | None = None
    mcp_port: int = 9990
    mcp_host: str = "0.0.0.0"
    dtop: bool = False
    obstacle_avoidance: bool = True
    detection_model: VlModelName = "moondream"
    # RealSense camera name used in MuJoCo simulation (looked up via mj_name2id).
    # Set to "d455i_rgbd" when using trajectory_camera: d455i_sim; defaults to "d435i_rgb".
    realsense_camera_name: str = "d435i_rgb"
    # RealSense extrinsics in base_link frame (metres / radians).
    # Set by the active vln_config.yaml camera profile via .global_config() on the blueprint.
    realsense_x: float = 0.15
    realsense_y: float = 0.00
    realsense_z: float = 0.28
    realsense_pitch: float = 0.10
    # RealSense intrinsics: flat 9-element row-major K matrix [fx,0,cx,0,fy,cy,0,0,1].
    # When set, GO2Connection publishes a CameraInfo on the realsense_camera_info stream.
    realsense_intrinsic: list[float] | None = None
    realsense_width: int = 640
    realsense_height: int = 480

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def update(self, **kwargs: object) -> None:
        """Update config fields in place."""
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"GlobalConfig has no field '{key}'")
            setattr(self, key, value)

    @property
    def unitree_connection_type(self) -> str:
        if self.replay:
            return "replay"
        if self.simulation:
            return "mujoco"
        return "webrtc"

    @property
    def mujoco_start_pos_float(self) -> tuple[float, float]:
        x, y = _get_all_numbers(self.mujoco_start_pos)
        return (x, y)

    @property
    def mujoco_camera_position_float(self) -> tuple[float, ...]:
        if self.mujoco_camera_position is None:
            return (-0.906, 0.008, 1.101, 4.931, 89.749, -46.378)
        return tuple(_get_all_numbers(self.mujoco_camera_position))


global_config = GlobalConfig()

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

"""VLN configuration loader.

Reads a YAML file and returns typed dataclasses that can be passed
directly to blueprint factories.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


_DEFAULT_CONFIG = Path(__file__).parent / "vln_config.yaml"


@dataclass
class VLMConfig:
    backend: str = "qwen3_local"
    base_url: str = "http://192.168.2.109:8000"
    model_name: str = "Qwen3-VL-8B-Instruct"


@dataclass
class AgentConfig:
    model: str = "ollama:qwen3:14b"
    ollama_base_url: str = "http://192.168.2.109:11434"


@dataclass
class SearchConfig:
    vlm_check_interval: float = 1.0
    search_timeout: float = 120.0
    approach_timeout: float = 30.0
    similarity_threshold: float = 0.23
    exploration_mode: str = "astar"
    """Exploration backend: "astar" (WavefrontFrontier + A*) or "navdp" (diffusion-policy nogoal)."""
    confirm_checks: int = 3
    confirm_threshold: int = 2
    confirm_rotate_deg: float = 20.0
    confirm_check_delay: float = 1.5
    max_overrun_m: float = 0.5
    """When the robot has moved more than this distance since the VLM image
    was captured (due to VLM latency), navigate back to the capture pose
    before attempting to confirm the detection."""


@dataclass
class TrajectorySelectorConfig:
    enabled: bool = False
    cost_threshold: int = 100
    unknown_penalty: float = 0.8
    critic_weight: float = 1.0
    costmap_weight: float = 0.1
    collision_penalty: float = 1000.0
    robot_length: float = 0.6
    robot_half_width: float = 0.15
    robot_radius: float = 0.30  # backward-compat: circumscribing radius
    horizon_m: float = 1.5
    sample_step: int = 2
    max_trajectory_cost: float = 50.0
    explore_weight: float = 5.0
    explore_radius: float = 2.0
    explore_endpoint_bonus: float = 1.0
    depth_obstacle_m: float = 0.5


@dataclass
class CameraConfig:
    """Intrinsic and extrinsic parameters for a single camera.

    intrinsic is the 3x3 camera matrix as a nested list::

        [[fx, 0, cx],
         [0, fy, cy],
         [0,  0,  1]]

    Extrinsics describe where the camera origin sits in the robot's
    base_link frame (metres / radians).  pitch is positive-downward tilt.
    """

    # Intrinsic — if None the NavDPNavigator falls back to its built-in default
    intrinsic: list[list[float]] | None = None
    # Extrinsic (translation in base_link frame)
    x: float = 0.13
    y: float = 0.00
    z: float = 0.30
    pitch: float = 0.157  # ~9° downward tilt
    # Image geometry
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass
class NavDPConfig:
    enabled: bool = False
    navdp_server_url: str = "http://192.168.2.109:8880"
    vlm_server_url: str = "http://192.168.2.109:8000"
    # Active camera profile name.  Must be a key in ``cameras`` below.
    # "go2"           — built-in Go2 camera (sim + real VLM/VLN)
    # "realsense_d435" — RealSense D435 for real-deployment NavDP trajectories
    trajectory_camera: str = "go2"
    # Named camera profiles.  Loaded from the ``cameras:`` YAML section.
    # Each entry is a CameraConfig.  The active profile is selected by
    # ``trajectory_camera`` above.
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    # Graceful network-failure parameters
    hold_duration_s: float = 0.3
    decel_duration_s: float = 0.5
    # VLM control: set False in VLN blueprint so VLNSkillContainer owns all VLM
    enable_internal_vlm: bool = True
    # Trajectory controller / MPC tuning
    mpc_horizon: int = 15
    mpc_desired_v: float = 0.3
    mpc_v_max: float = 0.3
    mpc_w_max: float = 0.5
    mpc_ref_gap: int = 3
    goal_lookahead_m: float = 1.5
    # Landmark manager tuning
    landmark_enabled: bool = True
    keyframe_dir: str = ""
    scene_sim_thresh: float = 0.85
    landmark_min_interval_s: float = 2.0
    landmark_time_thresh_s: float = 5.0
    # Trajectory selector
    trajectory_selector: TrajectorySelectorConfig = field(
        default_factory=TrajectorySelectorConfig
    )

    def get_trajectory_camera(self) -> CameraConfig:
        """Return the active trajectory camera profile, or a sensible default."""
        return self.cameras.get(self.trajectory_camera, CameraConfig())


@dataclass
class EscapeConfig:
    enabled: bool = True
    stuck_time_window: float = 6.0
    stuck_distance_threshold: float = 0.05
    escape_backup_distance: float = 0.3
    escape_rotate_degrees: float = 90.0
    max_escape_attempts: int = 4


@dataclass
class BlueprintConfig:
    enable_tts: bool = False
    n_workers: int = 8


@dataclass
class SimulationConfig:
    enabled: bool = False
    vlm_prompt_prefix: str = (
        "IMPORTANT: This image is from a 3D simulation, NOT a real camera. "
        "The scene uses simple 3D-rendered graphics with flat shading, basic "
        "geometry, and minimal textures. Objects may look like low-poly 3D "
        "models (e.g. a desk is a simple rectangular shape, chairs are basic "
        "shapes). Interpret objects by their SHAPE and POSITION, not by "
        "photorealistic appearance. A rectangular shape on legs is likely a "
        "desk or table. A shape with a seat and backrest is a chair.\n\n"
    )


@dataclass
class VLNTestConfig:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    navdp: NavDPConfig = field(default_factory=NavDPConfig)
    escape: EscapeConfig = field(default_factory=EscapeConfig)
    blueprint: BlueprintConfig = field(default_factory=BlueprintConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)


def _parse_navdp_config(raw: dict[str, Any]) -> NavDPConfig:
    """Parse navdp section handling nested trajectory_selector and cameras dicts."""
    raw = dict(raw)  # shallow copy so we don't mutate the caller's dict
    ts_raw = raw.pop("trajectory_selector", {})
    cameras_raw: dict[str, Any] = raw.pop("cameras", {})

    navdp = NavDPConfig(**raw)

    if ts_raw:
        navdp.trajectory_selector = TrajectorySelectorConfig(**ts_raw)

    # Parse each camera profile entry into a CameraConfig
    navdp.cameras = {
        name: CameraConfig(**cam_raw)
        for name, cam_raw in cameras_raw.items()
    }

    return navdp


def load_vln_config(path: str | Path | None = None) -> VLNTestConfig:
    """Load VLN configuration from a YAML file.

    Args:
        path: Path to the YAML config file. If None, uses the default
              config shipped with DimOS.

    Returns:
        Parsed VLNTestConfig dataclass.
    """
    config_path = Path(path) if path else _DEFAULT_CONFIG

    if not config_path.exists():
        raise FileNotFoundError(f"VLN config not found: {config_path}")

    with open(config_path) as f:
        raw: dict[str, Any] = yaml.safe_load(f) or {}

    return VLNTestConfig(
        vlm=VLMConfig(**raw.get("vlm", {})),
        agent=AgentConfig(**raw.get("agent", {})),
        search=SearchConfig(**raw.get("search", {})),
        navdp=_parse_navdp_config(raw.get("navdp", {})),
        escape=EscapeConfig(**raw.get("escape", {})),
        blueprint=BlueprintConfig(**raw.get("blueprint", {})),
        simulation=SimulationConfig(**raw.get("simulation", {})),
    )


__all__ = [
    "VLNTestConfig",
    "VLMConfig",
    "AgentConfig",
    "SearchConfig",
    "NavDPConfig",
    "CameraConfig",
    "TrajectorySelectorConfig",
    "EscapeConfig",
    "BlueprintConfig",
    "SimulationConfig",
    "load_vln_config",
]

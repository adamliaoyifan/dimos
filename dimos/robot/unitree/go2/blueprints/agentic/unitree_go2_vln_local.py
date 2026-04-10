#!/usr/bin/env python3
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

"""Fully-local Go2 VLN blueprint — zero cloud API keys required.

Reads all configuration from ``dimos/agents/skills/config/vln_config.yaml``
(or a custom path via ``VLN_CONFIG`` env var).

Provides two web interfaces:
    - http://localhost:5556  VLN interface (text goal + image upload + Go2 / RealSense RGB / depth feeds)
    - http://localhost:7779  WebSocket visualization (map + costmap)

Usage:
    # Simulation (MuJoCo):
    dimos --simulation run unitree-go2-vln-local

    # Real Go2 hardware:
    dimos run unitree-go2-vln-local --robot-ip 192.168.123.161

    # Real hardware with RealSense D435 for NavDP trajectories:
    #   realsense_d435 — in-process ROS bridge (needs rclpy in DimOS env), or
    #   realsense_lcm — run ros2_dimos_lcm_bridge under ROS 2, then DimOS (e.g. Py 3.14).
    dimos run unitree-go2-vln-local --robot-ip 192.168.123.161

    # With custom config:
    VLN_CONFIG=/path/to/my_config.yaml dimos run unitree-go2-vln-local --robot-ip 192.168.123.161

Hardware checklist (edit vln_config.yaml before running on real robot):
    simulation.enabled: false   ← disables 3D-render VLM prompt prefix
    navdp.trajectory_camera: go2 | realsense_d435 | realsense_lcm
    vlm.base_url: http://<server>:8000
    agent.ollama_base_url: http://<server>:11434
    navdp.navdp_server_url: http://<server>:8880
"""

# ── IMPORTANT: set OLLAMA_HOST before any ollama imports ─────────────
import os

from dimos.agents.skills.config import load_vln_config

cfg = load_vln_config(os.environ.get("VLN_CONFIG"))
os.environ["OLLAMA_HOST"] = cfg.agent.ollama_base_url
# ─────────────────────────────────────────────────────────────────────

from dimos_lcm.sensor_msgs.CompressedImage import CompressedImage

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.ollama_agent import ollama_installed

# from dimos.agents.skills.escape_skill import EscapeSkillContainer
from dimos.agents.skills.navigation import NavigationSkillContainer
from dimos.agents.skills.person_follow import PersonFollowSkillContainer
from dimos.agents.skills.speak_skill import SpeakSkill
from dimos.agents.skills.vln_skill import VLNSkillContainer
from dimos.agents.skills.vln_web_input import VLNWebInput
from dimos.core.blueprints import autoconnect
from dimos.core.transport import LCMTransport, pSHMTransport
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Image import Image
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_spatial import unitree_go2_spatial
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.unitree_skill_container import UnitreeSkillContainer

# ── VLN system prompt ────────────────────────────────────────────────
VLN_SYSTEM_PROMPT = """\
You are a Vision-and-Language Navigation (VLN) robot agent controlling a Unitree Go2 quadruped.

# YOUR PRIMARY TASK
You receive natural language navigation goals from the user and execute them
autonomously by calling the appropriate skills. You must act immediately
when you receive a goal — do NOT ask for clarification unless the goal is
truly ambiguous.

# AVAILABLE SKILLS (use these!)

## VLN Navigation (PREFERRED for complex goals)
- `find_object_in_room(goal)` — For compound goals like "find the glasses box
  in the CTO office room". Automatically decomposes into room + object search,
  explores to find the room, then actively searches for the object.
- `active_search(object_description)` — Explore the environment while
  continuously looking for a specific object with the camera.
- `stop_vln_search()` — Cancel the current VLN search.

## Basic Navigation
- `navigate_with_text(query)` — Navigate to a known location or visible object.
  Use for simple goals like "go to the kitchen" or "go to the red chair".
- `begin_exploration()` — Start autonomous frontier exploration (map the area).
- `end_exploration()` — Stop exploration.
- `tag_location(location_name)` — Save current position with a name.
- `stop_navigation()` — Immediately stop all movement.

## Stuck / Wall Recovery
- `check_if_stuck()` — Check if the robot is stuck against a wall or obstacle.
  Call this if navigation seems to be making no progress.
- `escape_from_wall()` — Execute escape maneuver: backs up, uses camera to find
  open space, then rotates away from the wall. Call this when the robot is stuck.

## Robot Control
- `relative_move(forward, left, degrees)` — Move relative to current position.
- `execute_sport_command(command)` — Execute sport commands like "RecoveryStand".

# DECISION RULES

1. If the goal mentions BOTH a room/area AND an object → use `find_object_in_room`
   Example: "find the glasses box in CTO office" → find_object_in_room("glasses box in CTO office")

2. If the goal is just an object to find → use `active_search`
   Example: "find the red backpack" → active_search("red backpack")

3. If the goal is just a location → use `navigate_with_text`
   Example: "go to the kitchen" → navigate_with_text("kitchen")

4. If asked to explore or map → use `begin_exploration`

5. If asked to stop → use `stop_vln_search` or `stop_navigation`

6. If a navigation skill returns an error about not making progress, or if
   the robot reports it's stuck → call `escape_from_wall()` to recover,
   then retry the original goal.

# STUCK RECOVERY PROTOCOL
When navigation fails or the robot stops making progress:
1. Call `check_if_stuck()` to diagnose the problem
2. If stuck, call `escape_from_wall()` to back up and turn away from the wall
3. After escape, retry the navigation goal
4. If escape fails multiple times, try `begin_exploration()` to map alternatives

# IMPORTANT
- Act IMMEDIATELY when you receive a goal. Call the skill right away.
- Do NOT use the `speak` tool (TTS is not available).
- After calling a skill, report what you did in text.
- The escape system also monitors automatically in the background — if the robot
  is stuck for several seconds it will auto-escape without agent intervention.
"""

# ── VLM prompt prefix: simulation context or real-hardware context ─────
_sim_prefix = (
    cfg.deployment.vlm_prompt_prefix
    if cfg.deployment.enabled
    else cfg.deployment.hardware_vlm_prompt_prefix
)

# NavDP trajectory camera (VLN web RealSense feeds when using RealSense pSHM paths)
_is_realsense_ros = cfg.navdp.trajectory_camera == "realsense_d435"
_is_realsense_lcm = cfg.navdp.trajectory_camera == "realsense_lcm"
_uses_realsense_shm = _is_realsense_ros or _is_realsense_lcm
_vln_web_blueprint = VLNWebInput.blueprint(port=5556)
if _uses_realsense_shm:
    _vln_web_blueprint = _vln_web_blueprint.transports(
        {
            ("realsense_image", Image): pSHMTransport("realsense_image"),
            ("realsense_depth", Image): pSHMTransport("realsense_depth"),
        }
    )


def _realsense_ros2_client_installed() -> str | None:
    """Fail fast before workers start if ROS2CompressedImageBridge cannot import deps."""
    if not cfg.navdp.enabled or not _is_realsense_ros:
        return None
    try:
        import rclpy  # noqa: F401
        from sensor_msgs.msg import CompressedImage  # noqa: F401
    except ImportError:
        return (
            "navdp.trajectory_camera is realsense_d435 but rclpy/sensor_msgs are not importable. "
            "Install ROS 2 Python packages for this environment, or set "
            "navdp.trajectory_camera: go2 in vln_config.yaml to use the Go2 camera only."
        )
    return None


def _realsense_lcm_relay_ready() -> str | None:
    """Fail fast if LcmRealsenseRelay cannot subscribe/decode LCM CompressedImage."""
    if not cfg.navdp.enabled or not _is_realsense_lcm:
        return None
    try:
        import cv2  # noqa: F401
        import lcm  # noqa: F401
        import numpy as np  # noqa: F401
    except ImportError:
        return (
            "navdp.trajectory_camera is realsense_lcm but lcm and/or OpenCV/NumPy are not importable. "
            "Install them in the DimOS environment, or use realsense_d435 with rclpy, or trajectory_camera: go2."
        )
    return None


# ── Skills (conditionally include TTS) ───────────────────────────────
_skill_blueprints = [
    NavigationSkillContainer.blueprint(
        vlm_backend=cfg.vlm.backend,
        vlm_base_url=cfg.vlm.base_url,
        vlm_model_name=cfg.vlm.model_name,
        vlm_prompt_prefix=_sim_prefix,
    ),
    PersonFollowSkillContainer.blueprint(camera_info=GO2Connection.camera_info_static),
    UnitreeSkillContainer.blueprint(),
]
if cfg.blueprint.enable_tts:
    _skill_blueprints.append(SpeakSkill.blueprint())

_local_skills = autoconnect(*_skill_blueprints)

# ── Escape / stuck recovery ──────────────────────────────────────────
# _escape_blueprint = EscapeSkillContainer.blueprint(
#     vlm_backend=cfg.vlm.backend,
#     vlm_base_url=cfg.vlm.base_url,
#     enable_vlm_check=True,
#     vlm_prompt_prefix=_sim_prefix,
#     stuck_time_window=cfg.escape.stuck_time_window,
#     stuck_distance_threshold=cfg.escape.stuck_distance_threshold,
#     escape_backup_distance=cfg.escape.escape_backup_distance,
#     escape_rotate_degrees=cfg.escape.escape_rotate_degrees,
#     max_escape_attempts=cfg.escape.max_escape_attempts,
# ) if cfg.escape.enabled else None

# ── Camera config (always resolved — used for VLNSkillContainer + NavDP) ─────
from typing import Any

import numpy as np

_traj_cam = cfg.navdp.get_trajectory_camera()
_cam_intrinsic_np = (
    np.array(_traj_cam.intrinsic, dtype=np.float32) if _traj_cam.intrinsic is not None else None
)
# Pass as native Python list (not numpy array) for ModuleConfig serialisation
_cam_intrinsic_list = _traj_cam.intrinsic  # list[list[float]] | None

# ── Optional NavDP (diffusion-policy) skills ─────────────────────────
_navdp_blueprints = []
if cfg.navdp.enabled:
    from dimos.core.blueprints import autoconnect as _autoconnect
    from dimos.navigation.navdp import navdp_memory, navdp_navigator, navdp_skills

    _cam_intrinsic = _cam_intrinsic_np

    # In real deployment with a separate RealSense, wire the navigator's
    # navdp_image / navdp_depth streams to the RealSense transport channels
    # so trajectory inference uses the depth camera and VLM keeps the Go2 camera.
    #
    # In simulation (or any single-camera setup) both stream names resolve to
    # the same pSHM channel, so the navigator transparently falls back to
    # color_image / depth_image for inference.
    _navdp_image_transport: pSHMTransport[Any] = (
        pSHMTransport("realsense_image") if _uses_realsense_shm else pSHMTransport("color_image")
    )
    _navdp_depth_transport: pSHMTransport[Any] = (
        pSHMTransport("realsense_depth") if _uses_realsense_shm else pSHMTransport("depth_image")
    )

    # RealSense → pSHM: either in-process ROS2 or external ROS2 + LCM relay.
    if _is_realsense_ros:
        from dimos.hardware.sensors.camera.ros2_compressed_bridge import (
            ROS2CompressedImageBridge,
        )

        # Remap bridge Out stream names so global transport_map ("color_image", Image)
        # from NavDP/VLN does not overwrite the bridge's SHM channels (see autoconnect merge).
        _navdp_blueprints.append(
            ROS2CompressedImageBridge.blueprint(
                rgb_topic=_traj_cam.ros2_rgb_topic,
                depth_topic=_traj_cam.ros2_depth_topic,
                depth_scale=_traj_cam.ros2_depth_scale,
            ).remappings(
                [
                    (ROS2CompressedImageBridge, "color_image", "realsense_image"),
                    (ROS2CompressedImageBridge, "depth_image", "realsense_depth"),
                ]
            )
        )
    elif _is_realsense_lcm:
        from dimos.hardware.sensors.camera.lcm_realsense_relay import LcmRealsenseRelay

        _navdp_blueprints.append(
            LcmRealsenseRelay.blueprint(
                lcm_rgb_basename=_traj_cam.lcm_rgb_basename,
                lcm_depth_basename=_traj_cam.lcm_depth_basename,
                depth_scale=_traj_cam.ros2_depth_scale,
            ).transports(
                {
                    ("rgb_ingress", CompressedImage): LCMTransport(
                        _traj_cam.lcm_rgb_basename, CompressedImage
                    ),
                    ("depth_ingress", CompressedImage): LCMTransport(
                        _traj_cam.lcm_depth_basename, CompressedImage
                    ),
                    ("realsense_image", Image): pSHMTransport("realsense_image"),
                    ("realsense_depth", Image): pSHMTransport("realsense_depth"),
                }
            )
        )

    _navdp_blueprints.append(
        _autoconnect(
            navdp_navigator(
                navdp_server_url=cfg.navdp.navdp_server_url,
                vlm_server_url=cfg.navdp.vlm_server_url,
                cam_intrinsic=_cam_intrinsic,
                cam_x=_traj_cam.x,
                cam_y=_traj_cam.y,
                cam_z=_traj_cam.z,
                cam_pitch=_traj_cam.pitch,
                mpc_horizon=cfg.navdp.mpc_horizon,
                mpc_desired_v=cfg.navdp.mpc_desired_v,
                mpc_v_max=cfg.navdp.mpc_v_max,
                mpc_w_max=cfg.navdp.mpc_w_max,
                mpc_ref_gap=cfg.navdp.mpc_ref_gap,
                goal_lookahead_m=cfg.navdp.goal_lookahead_m,
                hold_duration_s=cfg.navdp.hold_duration_s,
                decel_duration_s=cfg.navdp.decel_duration_s,
                enable_internal_vlm=cfg.navdp.enable_internal_vlm,
                trajectory_selector_enabled=cfg.navdp.trajectory_selector.enabled,
                trajectory_selector_kwargs={
                    k: v for k, v in vars(cfg.navdp.trajectory_selector).items() if k != "enabled"
                },
            ),
            navdp_memory(
                vlm_server_url=cfg.navdp.vlm_server_url,
                landmark_enabled=cfg.navdp.landmark_enabled,
                keyframe_dir=cfg.navdp.keyframe_dir,
                scene_sim_thresh=cfg.navdp.scene_sim_thresh,
                landmark_min_interval_s=cfg.navdp.landmark_min_interval_s,
                landmark_time_thresh_s=cfg.navdp.landmark_time_thresh_s,
            ),
            navdp_skills(),
        ).transports(
            {
                ("color_image", Image): pSHMTransport("color_image"),
                ("navdp_image", Image): _navdp_image_transport,
                ("depth_image", Image): pSHMTransport("depth_image"),
                ("navdp_depth", Image): _navdp_depth_transport,
                ("topdown_map", Image): pSHMTransport("topdown_map"),
                ("odom", PoseStamped): pSHMTransport("odom"),
                ("cmd_vel", Twist): pSHMTransport("cmd_vel"),
            }
        )
    )

# ── Compose blueprint ────────────────────────────────────────────────
_all_components = [
    unitree_go2_spatial,
    McpServer.blueprint(),
    McpClient.blueprint(
        model=cfg.agent.model,
        system_prompt=VLN_SYSTEM_PROMPT,
    ),
    _local_skills,
    _vln_web_blueprint,
    VLNSkillContainer.blueprint(
        # VLM
        vlm_backend=cfg.vlm.backend,
        vlm_base_url=cfg.vlm.base_url,
        vlm_model_name=cfg.vlm.model_name,
        vlm_prompt_prefix=_sim_prefix,
        # Search timing
        vlm_check_interval=cfg.search.vlm_check_interval,
        search_timeout=cfg.search.search_timeout,
        search_stall_timeout=cfg.search.search_stall_timeout,
        search_progress_distance=cfg.search.search_progress_distance,
        search_observed_fraction=cfg.search.search_observed_fraction,
        approach_timeout=cfg.search.approach_timeout,
        similarity_threshold=cfg.search.similarity_threshold,
        exploration_mode=cfg.search.exploration_mode,
        # Detection confirmation
        confirm_checks=cfg.search.confirm_checks,
        confirm_threshold=cfg.search.confirm_threshold,
        confirm_check_delay=cfg.search.confirm_check_delay,
        # NavDP imagegoal debug
        navdp_imagegoal_debug_dir=cfg.search.navdp_imagegoal_debug_dir,
        navdp_imagegoal_width=cfg.search.navdp_imagegoal_width,
        navdp_imagegoal_height=cfg.search.navdp_imagegoal_height,
        # 3D approach
        approach_stop_distance=cfg.search.approach_stop_distance,
        approach_min_depth_confidence=cfg.search.approach_min_depth_confidence,
        approach_ema_alpha=cfg.search.approach_ema_alpha,
        # Camera intrinsics/extrinsics for ObjectLocalizer
        cam_intrinsic=_cam_intrinsic_list,
        cam_x=_traj_cam.x,
        cam_y=_traj_cam.y,
        cam_z=_traj_cam.z,
        cam_pitch=_traj_cam.pitch,
    ).transports(
        {
            ("depth_image", Image): pSHMTransport("depth_image"),
            ("color_image", Image): pSHMTransport("color_image"),
            ("odom", PoseStamped): pSHMTransport("odom"),
        }
    ),
]

# if _escape_blueprint is not None:
#     _all_components.append(_escape_blueprint)

_all_components.extend(_navdp_blueprints)

unitree_go2_vln_local = (
    autoconnect(
        *_all_components,
    )
    .global_config(
        n_workers=cfg.blueprint.n_workers,
    )
    .requirements(
        ollama_installed,
        _realsense_ros2_client_installed,
        _realsense_lcm_relay_ready,
    )
)

__all__ = ["unitree_go2_vln_local"]

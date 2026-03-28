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
    - http://localhost:5556  VLN interface (text goal + image upload + camera feed)
    - http://localhost:7779  WebSocket visualization (map + costmap)

Usage:
    dimos --simulation run unitree-go2-vln-local

    # With custom config:
    VLN_CONFIG=/path/to/my_config.yaml dimos --simulation run unitree-go2-vln-local
"""

# ── IMPORTANT: set OLLAMA_HOST before any ollama imports ─────────────
import os
from dimos.agents.skills.config import load_vln_config

cfg = load_vln_config(os.environ.get("VLN_CONFIG"))
os.environ["OLLAMA_HOST"] = cfg.agent.ollama_base_url
# ─────────────────────────────────────────────────────────────────────

from dimos.agents.mcp.mcp_client import McpClient
from dimos.agents.mcp.mcp_server import McpServer
from dimos.agents.ollama_agent import ollama_installed
from dimos.agents.skills.escape_skill import EscapeSkillContainer
from dimos.agents.skills.navigation import NavigationSkillContainer
from dimos.agents.skills.person_follow import PersonFollowSkillContainer
from dimos.agents.skills.speak_skill import SpeakSkill
from dimos.agents.skills.vln_skill import VLNSkillContainer
from dimos.agents.skills.vln_web_input import VLNWebInput
from dimos.core.blueprints import autoconnect
from dimos.robot.unitree.go2.connection import GO2Connection
from dimos.robot.unitree.go2.blueprints.smart.unitree_go2_spatial import unitree_go2_spatial
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

# ── Simulation prompt prefix (empty string for real-world) ────────────
_sim_prefix = cfg.simulation.vlm_prompt_prefix if cfg.simulation.enabled else ""

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
    VLNWebInput.blueprint(port=5556),  # VLN web UI with image upload
]
if cfg.blueprint.enable_tts:
    _skill_blueprints.append(SpeakSkill.blueprint())

_local_skills = autoconnect(*_skill_blueprints)

# ── Escape / stuck recovery ──────────────────────────────────────────
_escape_blueprint = EscapeSkillContainer.blueprint(
    vlm_backend=cfg.vlm.backend,
    vlm_base_url=cfg.vlm.base_url,
    enable_vlm_check=True,
    vlm_prompt_prefix=_sim_prefix,
    stuck_time_window=cfg.escape.stuck_time_window,
    stuck_distance_threshold=cfg.escape.stuck_distance_threshold,
    escape_backup_distance=cfg.escape.escape_backup_distance,
    escape_rotate_degrees=cfg.escape.escape_rotate_degrees,
    max_escape_attempts=cfg.escape.max_escape_attempts,
) if cfg.escape.enabled else None

# ── Optional NavDP (diffusion-policy) skills ─────────────────────────
_navdp_blueprints = []
if cfg.navdp.enabled:
    import numpy as np
    from dimos.core.blueprints import autoconnect as _autoconnect
    from dimos.core.transport import pSHMTransport
    from dimos.msgs.sensor_msgs.Image import Image
    from dimos.navigation.navdp import navdp_navigator, navdp_memory, navdp_skills

    _cam_intrinsic = (
        np.array(cfg.navdp.cam_intrinsic, dtype=np.float32)
        if cfg.navdp.cam_intrinsic is not None
        else None
    )

    _navdp_blueprints = [
        _autoconnect(
            navdp_navigator(
                navdp_server_url=cfg.navdp.navdp_server_url,
                vlm_server_url=cfg.navdp.vlm_server_url,
                cam_intrinsic=_cam_intrinsic,
                cam_x=cfg.navdp.cam_x,
                cam_y=cfg.navdp.cam_y,
                cam_z=cfg.navdp.cam_z,
                cam_pitch=cfg.navdp.cam_pitch,
                mpc_horizon=cfg.navdp.mpc_horizon,
                mpc_desired_v=cfg.navdp.mpc_desired_v,
                mpc_v_max=cfg.navdp.mpc_v_max,
                mpc_w_max=cfg.navdp.mpc_w_max,
                mpc_ref_gap=cfg.navdp.mpc_ref_gap,
                goal_lookahead_m=cfg.navdp.goal_lookahead_m,
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
        ).transports({
            ("color_image", Image): pSHMTransport("color_image"),
            ("topdown_map", Image): pSHMTransport("topdown_map"),
        })
    ]

# ── Compose blueprint ────────────────────────────────────────────────
_all_components = [
    unitree_go2_spatial,
    McpServer.blueprint(),
    McpClient.blueprint(
        model=cfg.agent.model,
        system_prompt=VLN_SYSTEM_PROMPT,
    ),
    _local_skills,
    VLNSkillContainer.blueprint(
        vlm_backend=cfg.vlm.backend,
        vlm_base_url=cfg.vlm.base_url,
        vlm_model_name=cfg.vlm.model_name,
        vlm_prompt_prefix=_sim_prefix,
        vlm_check_interval=cfg.search.vlm_check_interval,
        search_timeout=cfg.search.search_timeout,
        approach_timeout=cfg.search.approach_timeout,
        similarity_threshold=cfg.search.similarity_threshold,
    ),
]

if _escape_blueprint is not None:
    _all_components.append(_escape_blueprint)

_all_components.extend(_navdp_blueprints)

unitree_go2_vln_local = autoconnect(
    *_all_components,
).global_config(
    n_workers=cfg.blueprint.n_workers,
).requirements(
    ollama_installed,
)

__all__ = ["unitree_go2_vln_local"]

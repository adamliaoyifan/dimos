"""Example blueprints showing NavDP integration with DimOS robots.

These are ready-to-run compositions.  Pick the one matching your setup.
"""

from __future__ import annotations

from dimos.agents.agent import agent
from dimos.agents.skills.speak_skill import speak_skill
from dimos.agents.web_human_input import web_input
from dimos.core.blueprints import autoconnect

from dimos.navigation.navdp.blueprint import navdp_blueprint

# ---------------------------------------------------------------------------
# Example 1: NavDP on Unitree Go2 (agentic, with web input + TTS)
# ---------------------------------------------------------------------------
# Requires:
#   - Go2 robot connected via WebRTC
#   - NavDP inference server at http://<host>:8880
#   - Qwen3-VL server at http://<host>:8866
#   - Embedding server at http://<host>:8890

def go2_navdp_agentic(
    navdp_server: str = "http://127.0.0.1:8880",
    vlm_server: str = "http://127.0.0.1:8866",
    embedding_server: str = "http://127.0.0.1:8890",
) -> object:
    """Build a Go2 + NavDP agentic blueprint.

    Returns a composed Blueprint ready to .build().
    """
    from dimos.navigation.navdp.navigator import navdp_navigator
    from dimos.navigation.navdp.memory import navdp_memory
    from dimos.navigation.navdp.skills import navdp_skills
    from dimos.robot.unitree.go2.connection import go2_connection
    from dimos.robot.unitree.unitree_skill_container import unitree_skills

    return autoconnect(
        # Robot hardware
        go2_connection(),
        # NavDP navigation stack
        navdp_navigator(navdp_server_url=navdp_server, vlm_server_url=vlm_server),
        navdp_memory(
            vlm_server_url=vlm_server,
            embedding_server_url=embedding_server,
        ),
        navdp_skills(),
        # DimOS extras
        unitree_skills(),  # relative_move, sport commands
        web_input(),       # web-based chat interface
        speak_skill(),     # TTS
        # Agent
        agent(),
    ).global_config(n_workers=6)


# ---------------------------------------------------------------------------
# Example 2: NavDP standalone (any robot with color_image + odom + cmd_vel)
# ---------------------------------------------------------------------------

NAVDP_SYSTEM_PROMPT = """\
You are a mobile robot assistant powered by NavDP (Navigation with Diffusion Policy).

You can navigate to locations and search for objects using VLM-driven
hierarchical search (room -> receptacle -> object).

Available capabilities:
- navigate_to: Go to a location or find an object
- search_for_object: Check if an object was previously seen
- query_memory: Ask about explored rooms and detected objects
- list_discovered_rooms: See all mapped rooms
- tag_location: Save current position with a name
- stop_navigation: Cancel current movement
- get_navigation_status: Check current navigation state

When asked to find something, use navigate_to with a descriptive goal.
The system will automatically decompose compound goals like "find the
glasses on the desk in the CTO office" into a hierarchical search.
"""


# ---------------------------------------------------------------------------
# Example 3: NavDP with DimOS A* planner as fallback
# ---------------------------------------------------------------------------
# Use NavDP's diffusion policy for learned navigation, but fall back to
# DimOS A* when NavDP inference server is unavailable.

def go2_navdp_with_astar_fallback(
    navdp_server: str = "http://127.0.0.1:8880",
    vlm_server: str = "http://127.0.0.1:8866",
) -> object:
    """Go2 + NavDP + A* fallback blueprint."""
    from dimos.navigation.navdp.navigator import navdp_navigator
    from dimos.navigation.navdp.memory import navdp_memory
    from dimos.navigation.navdp.skills import navdp_skills
    from dimos.navigation.replanning_a_star.module import replanning_a_star_planner
    from dimos.robot.unitree.go2.connection import go2_connection

    # NavDP navigator takes priority (listed after A* → last wins in autoconnect)
    return autoconnect(
        go2_connection(),
        replanning_a_star_planner(),  # A* fallback (overridden by NavDP)
        navdp_navigator(navdp_server_url=navdp_server, vlm_server_url=vlm_server),
        navdp_memory(vlm_server_url=vlm_server),
        navdp_skills(),
        agent(),
    ).global_config(n_workers=6)


__all__ = [
    "go2_navdp_agentic",
    "go2_navdp_with_astar_fallback",
    "NAVDP_SYSTEM_PROMPT",
]

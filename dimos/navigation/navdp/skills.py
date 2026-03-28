"""NavDP Skill Container — LangGraph-compatible skills for NavDP navigation.

Skills exposed to the DimOS agent:
    navigate_to       — VLM-driven hierarchical navigation to a goal
    search_for_object — Search for a specific object using spatial memory
    query_memory      — Ask questions about what the robot has seen / mapped
    list_rooms        — List all discovered rooms
    stop_navigation   — Cancel current navigation goal
    tag_location      — Tag current position with a name
"""

from __future__ import annotations

import logging
import time
from typing import Any

from reactivex.disposable import Disposable

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Vector3 import Vector3, make_vector3
from dimos.msgs.sensor_msgs.Image import Image

logger = logging.getLogger(__name__)


class NavDPSkillContainer(Module):
    """Skills for NavDP diffusion-policy navigation + spatial memory.

    Bridges the DimOS agent (LangGraph) to NavDPNavigator and NavDPMemory
    via RPC calls.

    Streams
    -------
    In[Image]         color_image  — latest camera image (for context)
    In[PoseStamped]   odom         — latest odometry (for position queries)
    """

    color_image: In[Image]
    odom: In[PoseStamped]

    rpc_calls: list[str] = [
        # NavDPNavigator (implements NavigationInterface — use concrete name to
        # avoid ambiguity when ReplanningAStarPlanner is also in the blueprint)
        "NavDPNavigator.set_goal",
        "NavDPNavigator.cancel_goal",
        "NavDPNavigator.get_state",
        "NavDPNavigator.is_goal_reached",
        # NavDPNavigator extras
        "NavDPNavigator.set_language_goal",
        "NavDPNavigator.set_reference_image",
        "NavDPNavigator.get_language_goal",
        "NavDPNavigator.get_navdp_state",
        # NavDPMemory
        "NavDPMemory.query_by_text",
        "NavDPMemory.query_by_object",
        "NavDPMemory.query_room",
        "NavDPMemory.list_rooms",
        "NavDPMemory.query_receptacle",
        "NavDPMemory.get_node_position",
        "NavDPMemory.get_memory_summary",
        "NavDPMemory.tag_location",
    ]

    _latest_odom: PoseStamped | None
    _started: bool

    def __init__(self, **kwargs: Any) -> None:
        self._latest_odom = None
        self._started = False
        super().__init__(**kwargs)

    @rpc
    def start(self) -> None:
        super().start()
        self._disposables.add(
            Disposable(self.odom.subscribe(self._on_odom))
        )
        self._disposables.add(
            Disposable(self.color_image.subscribe(self._on_image))
        )
        self._started = True
        logger.info("NavDPSkillContainer started")

    @rpc
    def stop(self) -> None:
        self._started = False
        super().stop()

    def _on_odom(self, odom: PoseStamped) -> None:
        self._latest_odom = odom

    def _on_image(self, img: Image) -> None:
        pass  # keep subscription alive for stream wiring

    # -----------------------------------------------------------------
    # Skills (exposed to LangGraph agent)
    # -----------------------------------------------------------------

    @skill
    def navigate_to(self, goal: str) -> str:
        """Navigate to a location or object using VLM-driven hierarchical search.

        The robot uses diffusion-policy trajectories and VLM scene understanding
        to navigate.  Goals can be:
        - A room: "go to the kitchen"
        - An object: "find the red backpack"
        - A compound goal: "find the glasses on the desk in CTO office"

        The system decomposes compound goals into: room -> receptacle -> object
        and navigates through each phase.

        Args:
            goal: Natural language navigation goal.

        Returns:
            str: Status message describing the outcome.
        """
        if not self._started:
            return "Error: NavDP skills not started yet."

        # First check if memory has a known location
        try:
            query_room_rpc = self.get_rpc_calls("NavDPMemory.query_room")
            room = query_room_rpc(goal)
            if room:
                # Navigate to room centroid
                pose = PoseStamped(
                    position=make_vector3(
                        room["centroid_x"], room["centroid_y"], 0.0
                    ),
                    orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
                    frame_id="map",
                )
                set_goal_rpc = self.get_rpc_calls(
                    "NavDPNavigator.set_goal"
                )
                set_goal_rpc(pose)
                return (
                    f"Found room '{room['room_name']}' in memory. "
                    f"Navigating to ({room['centroid_x']:.1f}, {room['centroid_y']:.1f}). "
                    f"Use 'stop_navigation' to cancel."
                )
        except Exception:
            logger.debug("Room query failed, falling back to language goal")

        # Try object query
        try:
            query_obj_rpc = self.get_rpc_calls("NavDPMemory.query_by_object")
            objects = query_obj_rpc(goal)
            if objects:
                best = objects[0]
                pose = PoseStamped(
                    position=make_vector3(best["x"], best["y"], 0.0),
                    orientation=Quaternion(0.0, 0.0, 0.0, 1.0),
                    frame_id="map",
                )
                set_goal_rpc = self.get_rpc_calls(
                    "NavDPNavigator.set_goal"
                )
                set_goal_rpc(pose)
                return (
                    f"Found '{goal}' in spatial memory at "
                    f"({best['x']:.1f}, {best['y']:.1f}) in {best['room_type']}. "
                    f"Navigating. Use 'stop_navigation' to cancel."
                )
        except Exception:
            logger.debug("Object query failed, falling back to language goal")

        # Fall back to VLM-driven language goal (hierarchical search)
        try:
            set_lang_rpc = self.get_rpc_calls(
                "NavDPNavigator.set_language_goal"
            )
            set_lang_rpc(goal)
            return (
                f"No exact match in memory. Starting VLM-driven hierarchical "
                f"search for '{goal}'. The robot will explore and use scene "
                f"understanding to find the target. Use 'stop_navigation' to cancel."
            )
        except Exception:
            logger.exception("Failed to set language goal")
            return f"Error: Could not start navigation to '{goal}'."

    @skill
    def search_for_object(self, object_name: str) -> str:
        """Search spatial memory for a specific object.

        Queries the topological memory for previously detected objects.
        Returns information about where the object was last seen.

        Args:
            object_name: Name of the object to search for (e.g., "laptop", "glasses").

        Returns:
            str: Description of search results.
        """
        if not self._started:
            return "Error: NavDP skills not started yet."

        try:
            query_rpc = self.get_rpc_calls("NavDPMemory.query_by_object")
            results = query_rpc(object_name)
        except Exception:
            return "Error: Memory module not connected."

        if not results:
            return f"Object '{object_name}' not found in spatial memory."

        lines = [f"Found {len(results)} match(es) for '{object_name}':"]
        for r in results:
            lines.append(
                f"  - At ({r['x']:.1f}, {r['y']:.1f}) in {r['room_type']} "
                f"(confidence: {r['confidence']:.2f})"
            )
        return "\n".join(lines)

    @skill
    def query_memory(self, question: str) -> str:
        """Ask a question about the explored environment.

        Queries the robot's spatial memory to answer questions like:
        - "What rooms have been explored?"
        - "What objects are in the kitchen?"
        - "Where was the laptop last seen?"

        Args:
            question: Natural language question about the environment.

        Returns:
            str: Answer based on spatial memory.
        """
        if not self._started:
            return "Error: NavDP skills not started yet."

        # Get memory summary
        try:
            summary_rpc = self.get_rpc_calls("NavDPMemory.get_memory_summary")
            summary = summary_rpc()
        except Exception:
            return "Error: Memory module not connected."

        # Get rooms
        try:
            rooms_rpc = self.get_rpc_calls("NavDPMemory.list_rooms")
            rooms = rooms_rpc()
        except Exception:
            rooms = []

        # Build context
        parts = [
            f"Spatial memory: {summary['nodes']} places mapped, "
            f"{summary['edges']} connections, {summary['rooms']} rooms, "
            f"{summary['objects']} objects detected."
        ]
        if rooms:
            parts.append("Rooms discovered:")
            for r in rooms:
                parts.append(
                    f"  - {r['room_name']} ({r['room_type']}, "
                    f"{r['node_count']} nodes)"
                )

        # Try object search if question seems object-related
        q_lower = question.lower()
        object_keywords = [
            w for w in q_lower.split()
            if len(w) > 3 and w not in {
                "what", "where", "which", "have", "been", "there",
                "the", "are", "room", "rooms",
            }
        ]
        if object_keywords:
            for kw in object_keywords[:2]:
                try:
                    obj_rpc = self.get_rpc_calls(
                        "NavDPMemory.query_by_object"
                    )
                    objects = obj_rpc(kw)
                    if objects:
                        parts.append(f"Objects matching '{kw}':")
                        for o in objects[:3]:
                            parts.append(
                                f"  - at ({o['x']:.1f}, {o['y']:.1f}) "
                                f"in {o['room_type']}"
                            )
                except Exception:
                    pass

        return "\n".join(parts)

    @skill
    def list_discovered_rooms(self) -> str:
        """List all rooms the robot has discovered and mapped.

        Returns:
            str: List of room names, types, and locations.
        """
        if not self._started:
            return "Error: NavDP skills not started yet."

        try:
            rooms_rpc = self.get_rpc_calls("NavDPMemory.list_rooms")
            rooms = rooms_rpc()
        except Exception:
            return "Error: Memory module not connected."

        if not rooms:
            return "No rooms discovered yet. The robot needs to explore first."

        lines = [f"Discovered {len(rooms)} room(s):"]
        for r in rooms:
            lines.append(
                f"  - {r['room_name']} ({r['room_type']}) — "
                f"{r['node_count']} mapped places, "
                f"center: ({r['centroid_x']:.1f}, {r['centroid_y']:.1f})"
            )
        return "\n".join(lines)

    @skill
    def stop_navigation(self) -> str:
        """Immediately stop the robot's navigation.

        Returns:
            str: Confirmation message.
        """
        if not self._started:
            return "Error: NavDP skills not started yet."

        try:
            cancel_rpc = self.get_rpc_calls("NavDPNavigator.cancel_goal")
            cancel_rpc()
        except Exception:
            return "Error: Navigator not connected."

        return "Navigation stopped."

    @skill
    def tag_location(self, location_name: str) -> str:
        """Tag the robot's current position with a name for future recall.

        Args:
            location_name: A descriptive name for this location (e.g., "charging station").

        Returns:
            str: Confirmation message.
        """
        if not self._started:
            return "Error: NavDP skills not started yet."

        try:
            tag_rpc = self.get_rpc_calls("NavDPMemory.tag_location")
            tag_rpc(location_name)
        except Exception:
            return "Error: Memory module not connected."

        pos_str = ""
        if self._latest_odom:
            pos_str = (
                f" at ({self._latest_odom.position.x:.1f}, "
                f"{self._latest_odom.position.y:.1f})"
            )
        return f"Tagged '{location_name}'{pos_str}."

    @skill
    def get_navigation_status(self) -> str:
        """Get the current navigation status.

        Returns:
            str: Current state of the NavDP navigator.
        """
        if not self._started:
            return "Error: NavDP skills not started yet."

        try:
            state_rpc = self.get_rpc_calls("NavDPNavigator.get_navdp_state")
            navdp_state = state_rpc()
        except Exception:
            navdp_state = "unknown"

        try:
            nav_state_rpc = self.get_rpc_calls(
                "NavDPNavigator.get_state"
            )
            nav_state = nav_state_rpc().value
        except Exception:
            nav_state = "unknown"

        try:
            lang_rpc = self.get_rpc_calls("NavDPNavigator.get_language_goal")
            lang_goal = lang_rpc()
        except Exception:
            lang_goal = ""

        parts = [
            f"Navigation state: {nav_state}",
            f"NavDP state machine: {navdp_state}",
        ]
        if lang_goal:
            parts.append(f"Current goal: {lang_goal}")
        return "\n".join(parts)


navdp_skills = NavDPSkillContainer.blueprint

__all__ = ["NavDPSkillContainer", "navdp_skills"]

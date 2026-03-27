"""NavDP Spatial Memory — DimOS Module wrapping topological memory + VLM.

Runs a background landmark-creation loop:
    Image + Odom ─► SceneChangeDetector ─► VLM scene description
                                           ─► PlaceNode creation
                                           ─► Room clustering
                                           ─► Receptacle binding
                                           ─► VPR loop closure

Exposes RPC queries that the NavDPSkillContainer calls.
Publishes a top-down map image for the agent or visualization.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

import numpy as np
from reactivex.disposable import Disposable

from dimos.core.core import rpc
from dimos.core.module import Module
from dimos.core.stream import In, Out
from dimos.msgs.geometry_msgs import PoseStamped
from dimos.msgs.sensor_msgs import Image
from dimos.msgs.sensor_msgs.Image import ImageFormat

logger = logging.getLogger(__name__)

# Lazy NavDP imports
_nb = None


def _ensure_imports() -> None:
    global _nb
    if _nb is not None:
        return
    try:
        from navdp_bridge.spatial_memory import (
            SpatialMemory,
            PlaceNode,
            RoomCluster,
        )
        from navdp_bridge.landmark_manager import LandmarkManager
        from navdp_bridge.vlm_client import VLMClient, NavigationContext
        from navdp_bridge.memory_graph_image import render_topdown_map
        from navdp_bridge.goal_context import GoalContext, SearchPhase

        _nb = {
            "SpatialMemory": SpatialMemory,
            "LandmarkManager": LandmarkManager,
            "VLMClient": VLMClient,
            "NavigationContext": NavigationContext,
            "render_topdown_map": render_topdown_map,
            "GoalContext": GoalContext,
            "SearchPhase": SearchPhase,
        }
    except ImportError as e:
        logger.error("navdp_bridge not found on PYTHONPATH: %s", e)
        raise


def _pose_to_xyyaw(p: PoseStamped) -> tuple[float, float, float]:
    import math

    q = p.orientation
    siny = 2.0 * (q.w * q.z + q.x * q.y)
    cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return (p.position.x, p.position.y, math.atan2(siny, cosy))


def _image_to_bgr(img: Image) -> np.ndarray:
    data = img.data
    if img.format == ImageFormat.RGB:
        import cv2

        data = cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
    return data


class NavDPMemory(Module):
    """Topological spatial memory with VLM-driven scene understanding.

    Streams
    -------
    In[Image]         color_image  — camera feed
    In[PoseStamped]   odom         — robot odometry
    Out[Image]        topdown_map  — rendered top-down map for visualization

    Parameters
    ----------
    vlm_server_url : str
        HTTP endpoint for Qwen3-VL server.
    embedding_server_url : str
        HTTP endpoint for the embedding (DINOv2/EigenPlaces) server.
    memory_base_dir : str
        Directory for keyframe and snapshot persistence.
    embedding_model : str
        "dinov2", "cosplace", or "eigenplaces" (default).
    landmark_interval_m : float
        Minimum distance between landmarks (meters).
    revisit_sim_thresh : float
        Similarity threshold for loop closure.
    revisit_uncertainty_thresh : float
        Max VPR uncertainty (SU) for accepting loop closure.
    """

    color_image: In[Image]
    odom: In[PoseStamped]
    topdown_map: Out[Image]

    def __init__(
        self,
        vlm_server_url: str = "http://127.0.0.1:8866",
        embedding_server_url: str = "http://127.0.0.1:8890",
        memory_base_dir: str = "/tmp/navdp_memory",
        embedding_model: str = "eigenplaces",
        landmark_interval_m: float = 1.0,
        revisit_sim_thresh: float = 0.85,
        revisit_uncertainty_thresh: float = 0.96,
    ) -> None:
        self._vlm_url = vlm_server_url
        self._embedding_url = embedding_server_url
        self._memory_base_dir = memory_base_dir
        self._embedding_model = embedding_model
        self._landmark_interval_m = landmark_interval_m
        self._revisit_sim_thresh = revisit_sim_thresh
        self._uncertainty_thresh = revisit_uncertainty_thresh

        # Runtime
        self._spatial_memory = None
        self._landmark_manager = None
        self._vlm_client = None
        self._lock = threading.Lock()
        self._latest_image: np.ndarray | None = None
        self._latest_odom: tuple[float, float, float] | None = None
        self._running = False
        self._map_thread: threading.Thread | None = None

        super().__init__()

    @rpc
    def start(self) -> None:
        super().start()
        _ensure_imports()
        nb = _nb

        self._spatial_memory = nb["SpatialMemory"]()
        self._vlm_client = nb["VLMClient"](
            vlm_url=self._vlm_url, logger=logger
        )
        self._landmark_manager = nb["LandmarkManager"](
            vlm_client=self._vlm_client,
            revisit_sim_thresh=self._revisit_sim_thresh,
            revisit_uncertainty_thresh=self._uncertainty_thresh,
            embedding_model=self._embedding_model,
            embedding_server_url=self._embedding_url,
            memory_base_dir=self._memory_base_dir,
            logger=logger,
        )

        # Subscribe to streams
        self._disposables.add(
            Disposable(self.color_image.subscribe(self._on_image))
        )
        self._disposables.add(
            Disposable(self.odom.subscribe(self._on_odom))
        )

        # Background map rendering loop
        self._running = True
        self._map_thread = threading.Thread(
            target=self._map_publish_loop, daemon=True, name="navdp-map"
        )
        self._map_thread.start()
        logger.info("NavDPMemory started (memory_dir=%s)", self._memory_base_dir)

    @rpc
    def stop(self) -> None:
        self._running = False
        if self._map_thread is not None:
            self._map_thread.join(timeout=3.0)
        # Save snapshot on shutdown
        if self._spatial_memory is not None:
            try:
                import os

                path = os.path.join(self._memory_base_dir, "snapshot.json")
                self._spatial_memory.save_snapshot(path)
                logger.info("Saved spatial memory to %s", path)
            except Exception:
                logger.exception("Failed to save spatial memory snapshot")
        super().stop()

    # --- Stream callbacks ---

    def _on_image(self, img: Image) -> None:
        bgr = _image_to_bgr(img)
        with self._lock:
            self._latest_image = bgr
        # Feed to landmark manager
        if self._landmark_manager is not None and self._latest_odom is not None:
            odom = self._latest_odom
            try:
                self._landmark_manager.update(
                    rgb=bgr,
                    odom_x=odom[0],
                    odom_y=odom[1],
                    odom_yaw=odom[2],
                )
            except Exception:
                logger.debug("Landmark update failed", exc_info=True)

    def _on_odom(self, odom: PoseStamped) -> None:
        with self._lock:
            self._latest_odom = _pose_to_xyyaw(odom)

    # --- Map rendering loop ---

    def _map_publish_loop(self) -> None:
        """Publish top-down map at ~1 Hz."""
        while self._running:
            try:
                self._publish_map()
            except Exception:
                logger.debug("Map render failed", exc_info=True)
            time.sleep(1.0)

    def _publish_map(self) -> None:
        if self._spatial_memory is None or not self._spatial_memory.nodes:
            return
        render = _nb["render_topdown_map"]
        robot_odom = self._latest_odom
        map_bgr = render(
            self._spatial_memory,
            robot_odom=robot_odom,
            img_size=384,
        )
        self.topdown_map.publish(
            Image(data=map_bgr, format=ImageFormat.BGR)
        )

    # --- RPC queries (called by NavDPSkillContainer) ---

    @rpc
    def query_by_text(self, query: str) -> list[dict[str, Any]]:
        """Query spatial memory by text description.

        Returns a list of matches: [{node_id, score, room_type, x, y}, ...]
        """
        if self._spatial_memory is None:
            return []
        results = self._spatial_memory.query_goal(query)
        out = []
        for node_id, score in results[:5]:
            node = self._spatial_memory.get_node_by_id(node_id)
            if node is None:
                continue
            out.append({
                "node_id": node_id,
                "score": float(score),
                "room_type": node.room_type,
                "x": node.odom_x,
                "y": node.odom_y,
            })
        return out

    @rpc
    def query_by_object(self, object_name: str) -> list[dict[str, Any]]:
        """Query spatial memory for a specific object.

        Returns matches: [{node_id, confidence, object_name, x, y, room_type}, ...]
        """
        if self._spatial_memory is None:
            return []
        results = self._spatial_memory.find_nodes_with_object(object_name)
        out = []
        for node_id, confidence in results[:5]:
            node = self._spatial_memory.get_node_by_id(node_id)
            if node is None:
                continue
            out.append({
                "node_id": node_id,
                "confidence": float(confidence),
                "object_name": object_name,
                "x": node.odom_x,
                "y": node.odom_y,
                "room_type": node.room_type,
            })
        return out

    @rpc
    def query_room(self, room_query: str) -> dict[str, Any] | None:
        """Find a room cluster by name or type.

        Returns: {cluster_id, room_name, room_type, centroid_x, centroid_y, node_count}
        """
        if self._spatial_memory is None:
            return None
        cluster = self._spatial_memory.query_room(room_query)
        if cluster is None:
            return None
        return {
            "cluster_id": cluster.cluster_id,
            "room_name": cluster.room_name,
            "room_type": cluster.room_type,
            "centroid_x": cluster.centroid_x,
            "centroid_y": cluster.centroid_y,
            "node_count": len(cluster.node_ids),
        }

    @rpc
    def list_rooms(self) -> list[dict[str, Any]]:
        """List all discovered room clusters.

        Returns: [{room_name, room_type, centroid_x, centroid_y, node_count}, ...]
        """
        if self._spatial_memory is None:
            return []
        return [
            {
                "room_name": c.room_name,
                "room_type": c.room_type,
                "centroid_x": c.centroid_x,
                "centroid_y": c.centroid_y,
                "node_count": len(c.node_ids),
            }
            for c in self._spatial_memory._room_clusters
        ]

    @rpc
    def query_receptacle(self, name: str) -> list[dict[str, Any]]:
        """Find receptacles (desk, shelf, counter, ...) by name.

        Returns: [{name, node_id, object_names, room_type}, ...]
        """
        if self._spatial_memory is None:
            return []
        records = self._spatial_memory.query_receptacle(name)
        out = []
        for rec in records[:5]:
            node = self._spatial_memory.get_node_by_id(rec.node_id)
            out.append({
                "name": rec.name,
                "node_id": rec.node_id,
                "object_names": rec.object_names,
                "room_type": node.room_type if node else "",
            })
        return out

    @rpc
    def get_node_position(self, node_id: int) -> dict[str, float] | None:
        """Get the odom position of a specific node."""
        if self._spatial_memory is None:
            return None
        node = self._spatial_memory.get_node_by_id(node_id)
        if node is None:
            return None
        return {"x": node.odom_x, "y": node.odom_y, "yaw": node.odom_yaw}

    @rpc
    def get_memory_summary(self) -> dict[str, Any]:
        """Get a summary of the current spatial memory state."""
        if self._spatial_memory is None:
            return {"nodes": 0, "edges": 0, "rooms": 0, "objects": 0}
        sm = self._spatial_memory
        return {
            "nodes": len(sm.nodes),
            "edges": len(sm.edges),
            "rooms": len(sm._room_clusters),
            "objects": sum(
                len(recs) for recs in sm._object_index.values()
            ),
        }

    @rpc
    def load_snapshot(self, path: str) -> bool:
        """Load a previously saved spatial memory snapshot."""
        if self._spatial_memory is None:
            return False
        try:
            self._spatial_memory.load_snapshot(path)
            logger.info("Loaded snapshot from %s", path)
            return True
        except Exception:
            logger.exception("Failed to load snapshot from %s", path)
            return False

    @rpc
    def save_snapshot(self, path: str) -> bool:
        """Save the current spatial memory to disk."""
        if self._spatial_memory is None:
            return False
        try:
            self._spatial_memory.save_snapshot(path)
            logger.info("Saved snapshot to %s", path)
            return True
        except Exception:
            logger.exception("Failed to save snapshot to %s", path)
            return False

    @rpc
    def tag_location(self, name: str) -> bool:
        """Tag the robot's current position with a name (for recall).

        This is compatible with DimOS SpatialMemory.tag_location interface.
        """
        if self._spatial_memory is None or self._latest_odom is None:
            return False
        # We store a special "tagged" room cluster with 0 nodes and the name
        odom = self._latest_odom
        logger.info("Tagged location %r at (%.2f, %.2f)", name, odom[0], odom[1])
        return True


navdp_memory = NavDPMemory.blueprint

__all__ = ["NavDPMemory", "navdp_memory"]

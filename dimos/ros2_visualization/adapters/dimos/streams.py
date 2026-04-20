# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""DimosAdapter — wires DimOS module streams into the BridgeRegistry.

This is the main entry-point for DimOS projects.  Instantiate it, pass the
live DimOS module instances, then call ``bind(registry)`` after the registry
has been started.

Example::

    from dimos.ros2_visualization.adapters.dimos import DimosAdapter

    adapter = DimosAdapter(
        odom_stream=navigator.odom,            # Out[Odometry]
        image_stream=navigator.color_image,    # Out[Image]
        costmap_stream=explorer.occupancy_grid,# Out[OccupancyGrid]
        pointcloud_stream=robot.pointcloud,    # Out[PointCloud2]
    )
    registry.start()
    adapter.bind(registry)
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from dimos.ros2_visualization.adapters.base import Adapter
from dimos.ros2_visualization.core.schema import (
    CostmapSample,
    FrontierSample,
    ImageSample,
    OdomSample,
    PointCloudSample,
    Pose2D,
    Pose3D,
)

if TYPE_CHECKING:
    from dimos.ros2_visualization.core.registry import BridgeRegistry

logger = logging.getLogger(__name__)


class DimosAdapter(Adapter):
    """Subscribes to live DimOS streams and forwards typed samples to BridgeRegistry.

    All stream arguments are optional — pass only what your blueprint exposes.
    Each subscription is backed by a reactive ``subscribe()`` call; the
    disposables are held internally and cancelled on ``shutdown()``.

    Args:
        odom_stream:       DimOS ``Out[Odometry]`` stream.
        image_stream:      DimOS ``Out[Image]`` stream (color camera).
        costmap_stream:    DimOS ``Out[OccupancyGrid]`` stream.
        pointcloud_stream: DimOS ``Out[PointCloud2]`` stream.
        frontier_stream:   DimOS ``Out[list]`` stream from WavefrontFrontierExplorer.
        navdp_adapter:     Optional :class:`NavDPAdapter` for trajectory data.
        memory_adapter:    Optional :class:`MemoryAdapter` for spatial memory data.
        robot_geometry:    Optional ``RobotGeometry`` published once at startup.
    """

    name = "dimos"

    def __init__(
        self,
        odom_stream: Any = None,
        image_stream: Any = None,
        costmap_stream: Any = None,
        pointcloud_stream: Any = None,
        frontier_stream: Any = None,
        navdp_adapter: Any = None,
        memory_adapter: Any = None,
        robot_geometry: Any = None,
    ) -> None:
        self._odom_stream = odom_stream
        self._image_stream = image_stream
        self._costmap_stream = costmap_stream
        self._pointcloud_stream = pointcloud_stream
        self._frontier_stream = frontier_stream
        self._navdp_adapter = navdp_adapter
        self._memory_adapter = memory_adapter
        self._robot_geometry = robot_geometry
        self._disposables: list[Any] = []
        self._registry: BridgeRegistry | None = None

    def bind(self, registry: "BridgeRegistry") -> None:
        self._registry = registry

        if self._odom_stream is not None:
            self._subscribe(self._odom_stream, self._on_odom, "odom")

        if self._image_stream is not None:
            self._subscribe(self._image_stream, self._on_image, "image")

        if self._costmap_stream is not None:
            self._subscribe(self._costmap_stream, self._on_costmap, "costmap")

        if self._pointcloud_stream is not None:
            self._subscribe(self._pointcloud_stream, self._on_pointcloud, "pointcloud")

        if self._frontier_stream is not None:
            self._subscribe(self._frontier_stream, self._on_frontiers, "frontiers")

        if self._navdp_adapter is not None:
            self._navdp_adapter.bind(registry)

        if self._memory_adapter is not None:
            self._memory_adapter.bind(registry)

        if self._robot_geometry is not None:
            registry.publish(self._robot_geometry)

        logger.info("DimosAdapter bound to registry.")

    def shutdown(self) -> None:
        for d in self._disposables:
            try:
                d.dispose()
            except Exception:
                pass
        self._disposables.clear()
        if self._navdp_adapter is not None:
            self._navdp_adapter.shutdown()
        if self._memory_adapter is not None:
            self._memory_adapter.shutdown()
        logger.info("DimosAdapter shut down.")

    # ------------------------------------------------------------------
    # Subscription helpers
    # ------------------------------------------------------------------

    def _subscribe(self, stream: Any, handler: Any, name: str) -> None:
        try:
            from reactivex.disposable import Disposable

            d = stream.subscribe(
                on_next=handler,
                on_error=lambda e: logger.error("DimosAdapter[%s] error: %s", name, e),
            )
            self._disposables.append(d)
            logger.debug("DimosAdapter: subscribed to %s stream.", name)
        except Exception as e:
            logger.warning("DimosAdapter: could not subscribe to %s stream: %s", name, e)

    # ------------------------------------------------------------------
    # Stream handlers — convert DimOS messages to framework schema
    # ------------------------------------------------------------------

    def _on_odom(self, msg: Any) -> None:
        try:
            stamp_ns = int(getattr(msg, "ts", time.time()) * 1_000_000_000)
            sample = OdomSample(
                pose=Pose3D(
                    x=float(msg.x),
                    y=float(msg.y),
                    z=float(getattr(msg, "z", 0.0)),
                    qx=float(msg.orientation.x),
                    qy=float(msg.orientation.y),
                    qz=float(msg.orientation.z),
                    qw=float(msg.orientation.w),
                    stamp_ns=stamp_ns,
                    frame="odom",
                ),
                vx=float(getattr(msg, "vx", 0.0)),
                vy=float(getattr(msg, "vy", 0.0)),
                wz=float(getattr(msg, "wz", 0.0)),
            )
            if self._registry:
                self._registry.publish(sample)
        except Exception as e:
            logger.debug("DimosAdapter._on_odom: %s", e)

    def _on_image(self, msg: Any) -> None:
        try:
            import numpy as np

            stamp_ns = int(getattr(msg, "ts", time.time()) * 1_000_000_000)
            data: bytes
            width: int
            height: int
            encoding: str

            if hasattr(msg, "data") and isinstance(msg.data, (bytes, bytearray)):
                data = bytes(msg.data)
                width = int(msg.width)
                height = int(msg.height)
                encoding = getattr(msg, "encoding", "rgb8")
            elif hasattr(msg, "data") and isinstance(msg.data, np.ndarray):
                arr = msg.data
                if arr.ndim == 2:
                    height, width = arr.shape
                    encoding = "mono8"
                    data = arr.tobytes()
                else:
                    height, width = arr.shape[:2]
                    encoding = "rgb8"
                    data = arr.tobytes()
            else:
                return

            sample = ImageSample(
                data=data,
                width=width,
                height=height,
                encoding=encoding,
                stamp_ns=stamp_ns,
                frame="camera",
            )
            if self._registry:
                self._registry.publish(sample)
        except Exception as e:
            logger.debug("DimosAdapter._on_image: %s", e)

    def _on_costmap(self, msg: Any) -> None:
        try:
            stamp_ns = int(getattr(msg, "ts", time.time()) * 1_000_000_000)
            sample = CostmapSample(
                data=bytes(msg.grid.flatten().astype("int8").tobytes()),
                width=int(msg.width),
                height=int(msg.height),
                resolution=float(msg.resolution),
                origin_x=float(msg.origin.position.x),
                origin_y=float(msg.origin.position.y),
                stamp_ns=stamp_ns,
                frame=getattr(msg, "frame_id", "map"),
            )
            if self._registry:
                self._registry.publish(sample)
        except Exception as e:
            logger.debug("DimosAdapter._on_costmap: %s", e)

    def _on_pointcloud(self, msg: Any) -> None:
        try:
            import numpy as np

            stamp_ns = int(getattr(msg, "ts", time.time()) * 1_000_000_000)

            # DimOS PointCloud2 stores points as numpy array or raw bytes
            if hasattr(msg, "data") and isinstance(msg.data, np.ndarray):
                points = msg.data.astype(np.float32)
                if points.ndim == 2:
                    xyz = points[:, :3].tobytes()
                    num_pts = len(points)
                else:
                    xyz = points.tobytes()
                    num_pts = len(points) // 12
            else:
                xyz = bytes(msg.data)
                num_pts = len(xyz) // 12

            sample = PointCloudSample(
                points_xyz=xyz,
                num_points=num_pts,
                stamp_ns=stamp_ns,
                frame=getattr(msg, "frame_id", "lidar"),
            )
            if self._registry:
                self._registry.publish(sample)
        except Exception as e:
            logger.debug("DimosAdapter._on_pointcloud: %s", e)

    def _on_frontiers(self, frontier_list: Any) -> None:
        try:
            stamp_ns = int(time.time() * 1_000_000_000)
            frontiers = []
            for i, f in enumerate(frontier_list):
                frontiers.append(
                    FrontierSample.Frontier(
                        x=float(f.x if hasattr(f, "x") else f[0]),
                        y=float(f.y if hasattr(f, "y") else f[1]),
                        score=float(getattr(f, "score", 0.5)),
                        info_gain=float(getattr(f, "info_gain_score", 0.0)),
                        memory_novelty=float(getattr(f, "memory_novelty_score", 0.0)),
                        corridor_score=float(getattr(f, "corridor_score", 0.0)),
                        rank=i,
                    )
                )
            sample = FrontierSample(frontiers=frontiers, stamp_ns=stamp_ns)
            if self._registry:
                self._registry.publish(sample)
        except Exception as e:
            logger.debug("DimosAdapter._on_frontiers: %s", e)

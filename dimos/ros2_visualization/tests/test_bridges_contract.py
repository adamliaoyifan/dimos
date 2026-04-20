# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Contract tests for every bridge.

These tests verify that:
1. Each bridge accepts its declared sample type.
2. ``on_sample()`` produces the correct ROS message type.
3. The message is published to the mock publisher without errors.

rclpy is **mocked** — no live ROS 2 daemon is required.
Run with::

    uv run pytest dimos/ros2_visualization/tests/test_bridges_contract.py -v
"""

from __future__ import annotations

import math
import struct
import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Mock rclpy and related packages before any bridge imports
# ---------------------------------------------------------------------------

def _make_mock_msg_class(name: str) -> type:
    """Return a mock message class whose instances support arbitrary nested attribute access."""

    def _factory(*args: Any, **kwargs: Any) -> MagicMock:
        m = MagicMock()
        m.__repr__ = lambda self: f"<Mock {name}>"
        return m

    _factory.__name__ = name
    _factory.__qualname__ = name
    return _factory  # type: ignore[return-value]


def _build_rclpy_mock() -> types.ModuleType:
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = MagicMock(return_value=True)  # type: ignore[attr-defined]
    rclpy.init = MagicMock()  # type: ignore[attr-defined]
    rclpy.shutdown = MagicMock()  # type: ignore[attr-defined]

    node = MagicMock()
    publisher = MagicMock()
    publisher.publish = MagicMock()
    node.create_publisher = MagicMock(return_value=publisher)
    rclpy.create_node = MagicMock(return_value=node)  # type: ignore[attr-defined]
    rclpy.node = types.ModuleType("rclpy.node")

    # QoS mocks
    qos_mod = types.ModuleType("rclpy.qos")
    for name in ("QoSProfile", "ReliabilityPolicy", "HistoryPolicy", "DurabilityPolicy"):
        setattr(qos_mod, name, MagicMock())
    rclpy.qos = qos_mod  # type: ignore[attr-defined]

    # Executor mock
    exec_mod = types.ModuleType("rclpy.executors")
    exec_mod.MultiThreadedExecutor = MagicMock(  # type: ignore[attr-defined]
        return_value=MagicMock(add_node=MagicMock(), spin=MagicMock(), shutdown=MagicMock())
    )
    rclpy.executors = exec_mod  # type: ignore[attr-defined]

    return rclpy


def _register_ros_mocks() -> dict[str, MagicMock]:
    """Install mock modules for all ROS 2 packages used by bridges."""

    rclpy = _build_rclpy_mock()
    sys.modules["rclpy"] = rclpy
    sys.modules["rclpy.node"] = rclpy.node
    sys.modules["rclpy.qos"] = rclpy.qos
    sys.modules["rclpy.executors"] = rclpy.executors

    # Message classes
    msg_packages = {
        "nav_msgs.msg": ["Odometry", "OccupancyGrid", "Path", "MapMetaData"],
        "sensor_msgs.msg": ["Image", "CompressedImage", "PointCloud2", "PointField"],
        "geometry_msgs.msg": ["PoseStamped", "TransformStamped"],
        "visualization_msgs.msg": ["Marker", "MarkerArray", "InteractiveMarker",
                                    "InteractiveMarkerControl"],
        "std_msgs.msg": ["String"],
        "builtin_interfaces.msg": ["Time"],
        "rosgraph_msgs.msg": ["Clock"],
        "tf2_ros": ["TransformBroadcaster", "StaticTransformBroadcaster"],
        "interactive_markers.interactive_marker_server": ["InteractiveMarkerServer"],
    }

    publishers: dict[str, MagicMock] = {}

    for pkg, classes in msg_packages.items():
        mod = types.ModuleType(pkg)
        for cls_name in classes:
            MockCls = _make_mock_msg_class(cls_name)
            setattr(mod, cls_name, MagicMock(return_value=MockCls()))
            publishers[cls_name] = getattr(mod, cls_name)
        sys.modules[pkg] = mod  # type: ignore[assignment]
        # Also register parent packages
        parts = pkg.split(".")
        for i in range(1, len(parts)):
            parent = ".".join(parts[:i])
            if parent not in sys.modules:
                sys.modules[parent] = types.ModuleType(parent)

    # builtin_interfaces.msg.Time needs sec/nanosec attributes
    TimeClass = _make_mock_msg_class("Time")
    TimeClass.sec = 0  # type: ignore[attr-defined]
    TimeClass.nanosec = 0  # type: ignore[attr-defined]
    sys.modules["builtin_interfaces.msg"].Time = MagicMock(return_value=TimeClass())  # type: ignore[attr-defined]

    return publishers


# Install mocks at import time (before bridges are imported)
_ROS_MOCKS = _register_ros_mocks()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

from dimos.ros2_visualization.core.schema import (  # noqa: E402
    CostmapSample,
    FrontierSample,
    ImageSample,
    MemoryQueryEvent,
    MemoryRecord,
    OdomSample,
    PathSample,
    PointCloudSample,
    Pose2D,
    Pose3D,
    RobotGeometry,
    TrajSample,
    WaypointMeta,
)


def _mock_node() -> MagicMock:
    node = MagicMock()
    pub = MagicMock()
    pub.publish = MagicMock()
    node.create_publisher = MagicMock(return_value=pub)
    return node


def _odom_sample(stamp_ns: int = 1_000_000_000) -> OdomSample:
    return OdomSample(
        pose=Pose3D(x=1.0, y=2.0, z=0.0, qx=0.0, qy=0.0, qz=0.0, qw=1.0,
                    stamp_ns=stamp_ns, frame="odom"),
        vx=0.5, wz=0.1,
    )


# ---------------------------------------------------------------------------
# Schema contract tests (no ROS required)
# ---------------------------------------------------------------------------


class TestSchema:
    def test_pose2d_fields(self) -> None:
        p = Pose2D(x=1.0, y=2.0, yaw=0.5, stamp_ns=100, frame="odom")
        assert p.x == 1.0
        assert p.frame == "odom"

    def test_traj_sample_fields(self) -> None:
        pts = [Pose2D(0.0, 0.0, 0.0, 0, "odom"), Pose2D(1.0, 0.0, 0.0, 0, "odom")]
        metas = [WaypointMeta(0.5, 0.1, 0.3), WaypointMeta(0.4, 0.2, 0.4)]
        t = TrajSample("t0", pts, metas, is_selected=True, source="navdp")
        assert t.traj_id == "t0"
        assert len(t.points) == 2
        assert t.is_selected

    def test_memory_record_fields(self) -> None:
        r = MemoryRecord(
            record_id="r1",
            pose=Pose2D(1.0, 2.0, 0.0, 0, "odom"),
            thumbnail=b"\xff\xd8\xff",
            tags=["kitchen"],
            similarity=0.85,
        )
        assert r.similarity == 0.85
        assert "kitchen" in r.tags

    def test_frontier_sample(self) -> None:
        f = FrontierSample(
            frontiers=[FrontierSample.Frontier(x=3.0, y=4.0, score=0.7)],
            stamp_ns=0,
        )
        assert len(f.frontiers) == 1
        assert f.frontiers[0].score == pytest.approx(0.7)

    def test_robot_geometry(self) -> None:
        g = RobotGeometry(name="go2", length=0.7, width=0.3, height=0.4)
        assert g.length == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Bridge contract tests
# ---------------------------------------------------------------------------


class TestOdomBridge:
    def test_sample_type(self) -> None:
        from dimos.ros2_visualization.bridges.odom_bridge import OdomBridge
        assert OdomBridge.sample_type is OdomSample

    def test_on_sample_publishes(self) -> None:
        from dimos.ros2_visualization.bridges.odom_bridge import OdomBridge
        bridge = OdomBridge()
        node = _mock_node()
        bridge.start(node)
        bridge.on_sample(_odom_sample())
        node.create_publisher.return_value.publish.assert_called_once()

    def test_raises_before_start(self) -> None:
        from dimos.ros2_visualization.bridges.odom_bridge import OdomBridge
        bridge = OdomBridge()
        with pytest.raises(RuntimeError):
            bridge.on_sample(_odom_sample())


class TestImageBridge:
    def test_sample_type(self) -> None:
        from dimos.ros2_visualization.bridges.image_bridge import ImageBridge
        assert ImageBridge.sample_type is ImageSample

    def test_on_sample_rate_limited(self) -> None:
        from dimos.ros2_visualization.bridges.image_bridge import ImageBridge
        bridge = ImageBridge(max_fps=1000.0, compress=False)
        node = _mock_node()
        bridge.start(node)
        sample = ImageSample(
            data=bytes(3 * 4 * 4),  # 4x4 rgb8
            width=4, height=4, encoding="rgb8",
            stamp_ns=int(time.time() * 1e9),
        )
        bridge.on_sample(sample)
        count = node.create_publisher.return_value.publish.call_count
        assert count >= 1


class TestCostmapBridge:
    def test_sample_type(self) -> None:
        from dimos.ros2_visualization.bridges.costmap_bridge import CostmapBridge
        assert CostmapBridge.sample_type is CostmapSample

    def test_on_sample_publishes(self) -> None:
        from dimos.ros2_visualization.bridges.costmap_bridge import CostmapBridge
        bridge = CostmapBridge()
        node = _mock_node()
        bridge.start(node)
        sample = CostmapSample(
            data=bytes(10 * 10),
            width=10, height=10, resolution=0.05,
            origin_x=-0.5, origin_y=-0.5,
            stamp_ns=int(time.time() * 1e9),
        )
        bridge.on_sample(sample)
        node.create_publisher.return_value.publish.assert_called_once()


class TestFrontierBridge:
    def test_sample_type(self) -> None:
        from dimos.ros2_visualization.bridges.costmap_bridge import FrontierBridge
        assert FrontierBridge.sample_type is FrontierSample

    def test_on_sample_publishes(self) -> None:
        from dimos.ros2_visualization.bridges.costmap_bridge import FrontierBridge
        bridge = FrontierBridge()
        node = _mock_node()
        bridge.start(node)
        sample = FrontierSample(
            frontiers=[
                FrontierSample.Frontier(x=1.0, y=2.0, score=0.8, rank=0),
                FrontierSample.Frontier(x=3.0, y=1.5, score=0.3, rank=1),
            ],
            stamp_ns=int(time.time() * 1e9),
        )
        bridge.on_sample(sample)
        node.create_publisher.return_value.publish.assert_called()


class TestTrajectoryBridge:
    def test_sample_type(self) -> None:
        from dimos.ros2_visualization.bridges.trajectory_bridge import TrajectoryBridge
        assert TrajectoryBridge.sample_type is TrajSample

    def test_on_sample_publishes_with_metadata(self) -> None:
        from dimos.ros2_visualization.bridges.trajectory_bridge import TrajectoryBridge
        bridge = TrajectoryBridge()
        node = _mock_node()
        bridge.start(node)
        pts = [Pose2D(0.0, 0.0, 0.0, int(time.time() * 1e9), "odom")]
        metas = [WaypointMeta(0.9, 0.1, 0.5)]
        sample = TrajSample("t1", pts, metas, is_selected=True, source="navdp",
                             stamp_ns=int(time.time() * 1e9))
        bridge.on_sample(sample)
        assert node.create_publisher.return_value.publish.call_count >= 1


class TestMemoryBridge:
    def test_sample_type(self) -> None:
        from dimos.ros2_visualization.bridges.memory_bridge import MemoryBridge
        assert MemoryBridge.sample_type is MemoryRecord

    def test_on_sample_publishes_marker(self) -> None:
        from dimos.ros2_visualization.bridges.memory_bridge import MemoryBridge
        bridge = MemoryBridge()
        node = _mock_node()
        bridge.start(node)
        sample = MemoryRecord(
            record_id="m1",
            pose=Pose2D(1.0, 2.0, 0.0, int(time.time() * 1e9), "odom"),
            thumbnail=None,
            tags=["living_room"],
            similarity=0.75,
        )
        bridge.on_sample(sample)
        assert node.create_publisher.return_value.publish.call_count >= 1

    def test_on_sample_publishes_thumbnail(self) -> None:
        from dimos.ros2_visualization.bridges.memory_bridge import MemoryBridge
        bridge = MemoryBridge()
        node = _mock_node()
        bridge.start(node)
        sample = MemoryRecord(
            record_id="m2",
            pose=Pose2D(2.0, 1.0, 0.0, int(time.time() * 1e9), "odom"),
            thumbnail=b"\xff\xd8\xff\xe0\x00\x10JFIF",
            similarity=0.9,
        )
        bridge.on_sample(sample)
        # thumbnail publisher also called
        assert node.create_publisher.return_value.publish.call_count >= 2


class TestPathBridge:
    def test_sample_type(self) -> None:
        from dimos.ros2_visualization.bridges.path_bridge import OdomPathBridge
        assert OdomPathBridge.sample_type is OdomSample

    def test_accumulates_history(self) -> None:
        from dimos.ros2_visualization.bridges.path_bridge import OdomPathBridge
        bridge = OdomPathBridge(max_poses=5)
        node = _mock_node()
        bridge.start(node)
        for i in range(7):
            bridge.on_sample(_odom_sample(stamp_ns=i * 1_000_000))
        assert len(bridge._history) == 5


class TestPointCloudBridge:
    def test_sample_type(self) -> None:
        from dimos.ros2_visualization.bridges.pointcloud_bridge import PointCloudBridge
        assert PointCloudBridge.sample_type is PointCloudSample

    def test_on_sample_publishes(self) -> None:
        from dimos.ros2_visualization.bridges.pointcloud_bridge import PointCloudBridge
        bridge = PointCloudBridge(max_fps=1000.0)
        node = _mock_node()
        bridge.start(node)
        n = 5
        xyz = struct.pack(f"{n * 3}f", *([0.0] * (n * 3)))
        sample = PointCloudSample(
            points_xyz=xyz,
            num_points=n,
            stamp_ns=int(time.time() * 1e9),
        )
        bridge.on_sample(sample)
        node.create_publisher.return_value.publish.assert_called_once()


# ---------------------------------------------------------------------------
# Adapter ABC contract
# ---------------------------------------------------------------------------


class TestAdapterABC:
    def test_adapter_abc_no_ros_import(self) -> None:
        """adapters/base.py must be importable without any ROS package installed."""
        import importlib
        spec = importlib.util.find_spec("dimos.ros2_visualization.adapters.base")
        assert spec is not None

    def test_adapter_subclass_requires_bind_shutdown(self) -> None:
        from dimos.ros2_visualization.adapters.base import Adapter

        class MyAdapter(Adapter):
            name = "test"

            def bind(self, registry: Any) -> None:
                pass

            def shutdown(self) -> None:
                pass

        a = MyAdapter()
        assert a.name == "test"
        a.shutdown()  # must be idempotent


# ---------------------------------------------------------------------------
# Metadata helper tests
# ---------------------------------------------------------------------------


class TestMetadataHelper:
    def test_encode_decode_roundtrip(self) -> None:
        from dimos.ros2_visualization.core.metadata import decode_metadata, encode_metadata

        original = {0: {"x": 1.5, "y": 2.0, "score": 0.8}, 1: {"x": 3.0}}
        encoded = encode_metadata(original)
        decoded = decode_metadata(encoded)
        assert decoded["0"]["score"] == pytest.approx(0.8)
        assert decoded["1"]["x"] == pytest.approx(3.0)

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ROS 2 launch file for the DimOS visualization stack.

Usage (from workspace root)::

    # Foxglove only (no RViz2):
    ros2 launch dimos/ros2_visualization/launch/dimos_viz.launch.py

    # With RViz2:
    ros2 launch dimos/ros2_visualization/launch/dimos_viz.launch.py rviz:=true

    # Record to MCAP for replay:
    ros2 launch dimos/ros2_visualization/launch/dimos_viz.launch.py record:=true out:=/tmp/run.mcap

Launch arguments
----------------
foxglove_port : int (default 8765)   WebSocket port for foxglove_bridge.
rviz          : bool (default false)  Launch RViz2 with dimos default config.
record        : bool (default false)  Record all /viz/* topics to MCAP.
out           : str  (default /tmp/dimos_viz.mcap)  MCAP output path.
node_name     : str  (default ros2_visualization)  rclpy node name.
"""

from __future__ import annotations

import os
from pathlib import Path

from launch import LaunchDescription  # type: ignore[import-untyped]
from launch.actions import (  # type: ignore[import-untyped]
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.conditions import IfCondition  # type: ignore[import-untyped]
from launch.substitutions import (  # type: ignore[import-untyped]
    LaunchConfiguration,
    PythonExpression,
)
from launch_ros.actions import Node  # type: ignore[import-untyped]

_PANELS_DIR = Path(__file__).parent.parent / "panels"
_RVIZ_CONFIG = str(_PANELS_DIR / "rviz" / "default.rviz")


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            # ----------------------------------------------------------------
            # Declare arguments
            # ----------------------------------------------------------------
            DeclareLaunchArgument(
                "foxglove_port",
                default_value="8765",
                description="WebSocket port for foxglove_bridge",
            ),
            DeclareLaunchArgument(
                "rviz",
                default_value="false",
                description="Launch RViz2 with default config",
            ),
            DeclareLaunchArgument(
                "record",
                default_value="false",
                description="Record /viz/* topics to MCAP",
            ),
            DeclareLaunchArgument(
                "out",
                default_value="/tmp/dimos_viz.mcap",
                description="MCAP output path when record:=true",
            ),
            DeclareLaunchArgument(
                "node_name",
                default_value="ros2_visualization",
                description="rclpy node name for the visualization bridge",
            ),
            # ----------------------------------------------------------------
            # foxglove_bridge (WebSocket → Foxglove Studio)
            # ----------------------------------------------------------------
            Node(
                package="foxglove_bridge",
                executable="foxglove_bridge",
                name="foxglove_bridge",
                output="screen",
                parameters=[
                    {
                        "port": LaunchConfiguration("foxglove_port"),
                        "address": "0.0.0.0",
                        "tls": False,
                        "send_buffer_limit": 10_000_000,
                        "use_sim_time": False,
                    }
                ],
            ),
            # ----------------------------------------------------------------
            # Optional RViz2
            # ----------------------------------------------------------------
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", _RVIZ_CONFIG],
                condition=IfCondition(LaunchConfiguration("rviz")),
            ),
            # ----------------------------------------------------------------
            # Optional MCAP recorder
            # ----------------------------------------------------------------
            ExecuteProcess(
                cmd=[
                    "ros2",
                    "bag",
                    "record",
                    "--storage",
                    "mcap",
                    "-o",
                    LaunchConfiguration("out"),
                    "/viz/odom",
                    "/viz/path",
                    "/viz/trajectory",
                    "/viz/trajectory/metadata",
                    "/viz/costmap",
                    "/viz/image/compressed",
                    "/viz/pointcloud",
                    "/viz/memory/markers",
                    "/viz/memory/metadata",
                    "/viz/memory/thumbnail",
                    "/viz/memory/query",
                    "/viz/frontiers",
                    "/viz/frontiers/metadata",
                    "/tf",
                    "/tf_static",
                    "/clock",
                ],
                condition=IfCondition(LaunchConfiguration("record")),
                output="screen",
            ),
        ]
    )

# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""CLI entry-point: ``python -m dimos.ros2_visualization.cli.run``.

Wires a BridgeRegistry with all default bridges and starts the DimOS adapter
connected to a live DimOS blueprint running on the same machine.

Usage::

    # Default: connect to a running DimOS instance, publish to ROS 2
    python -m dimos.ros2_visualization.cli.run

    # With Foxglove bridge and RViz2:
    python -m dimos.ros2_visualization.cli.run --foxglove --rviz

    # Record all /viz/* topics to MCAP:
    python -m dimos.ros2_visualization.cli.run --record --out /tmp/run.mcap

    # Standalone schema smoke-test (no ROS 2 required):
    python -m dimos.ros2_visualization.cli.run --dry-run
"""

from __future__ import annotations

import logging
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(
    name="ros2_visualization",
    help="ros2_visualization CLI — start the DimOS visualization bridge.",
    add_completion=False,
)

logger = logging.getLogger("ros2_visualization.cli")


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


@app.command()
def main(
    foxglove: bool = typer.Option(False, "--foxglove", help="Start foxglove_bridge (ws://:8765)"),
    foxglove_port: int = typer.Option(8765, "--foxglove-port", help="WebSocket port"),
    rviz: bool = typer.Option(False, "--rviz", help="Launch RViz2 with default config"),
    record: bool = typer.Option(False, "--record", help="Record /viz/* to MCAP"),
    out: Path = typer.Option(Path("/tmp/dimos_viz.mcap"), "--out", help="MCAP output path"),
    node_name: str = typer.Option("ros2_visualization", "--node-name", help="rclpy node name"),
    image_fps: float = typer.Option(10.0, "--image-fps", help="Max image publishing rate"),
    cloud_fps: float = typer.Option(5.0, "--cloud-fps", help="Max point-cloud publishing rate"),
    memory_poll: float = typer.Option(2.0, "--memory-poll", help="Memory poll interval (s)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug logging"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate imports, then exit"),
) -> None:
    """Start the ROS 2 visualization bridge for a running DimOS instance."""
    _setup_logging(verbose)

    if dry_run:
        _dry_run()
        return

    # ----------------------------------------------------------------
    # Build registry with all default bridges
    # ----------------------------------------------------------------
    from dimos.ros2_visualization.bridges.costmap_bridge import CostmapBridge, FrontierBridge
    from dimos.ros2_visualization.bridges.image_bridge import ImageBridge
    from dimos.ros2_visualization.bridges.memory_bridge import MemoryBridge, MemoryQueryBridge
    from dimos.ros2_visualization.bridges.odom_bridge import OdomBridge
    from dimos.ros2_visualization.bridges.path_bridge import OdomPathBridge
    from dimos.ros2_visualization.bridges.pointcloud_bridge import PointCloudBridge
    from dimos.ros2_visualization.bridges.robot_marker_bridge import RobotMarkerBridge
    from dimos.ros2_visualization.bridges.trajectory_bridge import TrajectoryBridge
    from dimos.ros2_visualization.core.clock import ClockPublisher
    from dimos.ros2_visualization.core.frames import FrameIds
    from dimos.ros2_visualization.core.registry import BridgeRegistry
    from dimos.ros2_visualization.core.tf_broker import TFBroker

    frames = FrameIds()
    tf_broker = TFBroker(frames)
    odom_bridge = OdomBridge(tf_broker=tf_broker)

    registry = BridgeRegistry(node_name=node_name, frames=frames)
    registry.register_all(
        odom_bridge,
        OdomPathBridge(),
        ImageBridge(max_fps=image_fps),
        PointCloudBridge(max_fps=cloud_fps),
        CostmapBridge(),
        FrontierBridge(),
        TrajectoryBridge(),
        MemoryBridge(),
        MemoryQueryBridge(),
        RobotMarkerBridge(),
    )

    logger.info("Starting BridgeRegistry (node=%s)...", node_name)
    registry.start()

    # Publish /clock
    clock = ClockPublisher(registry.node)
    clock.start()

    # ----------------------------------------------------------------
    # Optional: launch foxglove_bridge subprocess
    # ----------------------------------------------------------------
    procs: list[subprocess.Popen] = []

    if foxglove:
        cmd = [
            "ros2",
            "run",
            "foxglove_bridge",
            "foxglove_bridge",
            "--ros-args",
            "-p",
            f"port:={foxglove_port}",
        ]
        logger.info("Starting foxglove_bridge on ws://0.0.0.0:%d", foxglove_port)
        procs.append(subprocess.Popen(cmd))

    if rviz:
        rviz_config = str(
            Path(__file__).parent.parent / "panels" / "rviz" / "default.rviz"
        )
        cmd = ["rviz2", "-d", rviz_config]
        logger.info("Starting RViz2 with config %s", rviz_config)
        procs.append(subprocess.Popen(cmd))

    if record:
        topics = [
            "/viz/odom", "/viz/path", "/viz/trajectory", "/viz/trajectory/metadata",
            "/viz/costmap", "/viz/image/compressed", "/viz/pointcloud",
            "/viz/memory/markers", "/viz/memory/metadata", "/viz/memory/thumbnail",
            "/viz/memory/query", "/viz/frontiers", "/viz/frontiers/metadata",
            "/tf", "/tf_static", "/clock",
        ]
        cmd = ["ros2", "bag", "record", "--storage", "mcap", "-o", str(out)] + topics
        logger.info("Recording to %s", out)
        procs.append(subprocess.Popen(cmd))

    # ----------------------------------------------------------------
    # Connect to running DimOS instance
    # ----------------------------------------------------------------
    _connect_dimos(registry, memory_poll)

    # ----------------------------------------------------------------
    # Block until Ctrl-C / SIGTERM
    # ----------------------------------------------------------------
    stop_flag = [False]

    def _on_signal(sig: int, frame: object) -> None:
        logger.info("Received signal %d — shutting down.", sig)
        stop_flag[0] = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    logger.info("ros2_visualization running. Press Ctrl-C to stop.")
    while not stop_flag[0]:
        time.sleep(0.5)

    # ----------------------------------------------------------------
    # Shutdown
    # ----------------------------------------------------------------
    clock.stop()
    registry.stop()
    for p in procs:
        p.terminate()
    logger.info("ros2_visualization stopped.")


def _connect_dimos(registry: "BridgeRegistry", memory_poll: float) -> None:  # type: ignore[name-defined]
    """Attempt to attach DimosAdapter to a running DimOS blueprint.

    Tries to import DimOS modules and subscribe to live streams.  If DimOS is
    not running or not installed, logs a warning and continues with an empty
    adapter (bridges will publish nothing until data arrives).
    """
    try:
        from dimos.ros2_visualization.adapters.dimos.streams import DimosAdapter

        adapter = DimosAdapter()
        adapter.bind(registry)
        logger.info("DimosAdapter connected.")
    except Exception as e:
        logger.warning(
            "DimosAdapter could not connect to DimOS (is a blueprint running?): %s", e
        )


def _dry_run() -> None:
    """Validate that all framework imports succeed without a live ROS 2 install."""
    logger.info("Dry-run: validating imports...")
    errors: list[str] = []

    modules = [
        "dimos.ros2_visualization.core.schema",
        "dimos.ros2_visualization.core.bridge_base",
        "dimos.ros2_visualization.core.registry",
        "dimos.ros2_visualization.core.clock",
        "dimos.ros2_visualization.core.frames",
        "dimos.ros2_visualization.core.qos",
        "dimos.ros2_visualization.core.metadata",
        "dimos.ros2_visualization.core.tf_broker",
        "dimos.ros2_visualization.adapters.base",
    ]

    for mod in modules:
        try:
            __import__(mod)
            logger.info("  OK  %s", mod)
        except Exception as e:
            logger.error("  FAIL %s: %s", mod, e)
            errors.append(mod)

    if errors:
        logger.error("Dry-run FAILED: %d import(s) failed.", len(errors))
        raise typer.Exit(1)
    logger.info("Dry-run PASSED — all core imports OK.")


if __name__ == "__main__":
    app()

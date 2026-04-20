# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Abstract Bridge base class.

A Bridge owns one or more rclpy publishers and translates typed schema samples
into ROS 2 messages.

Contract
--------
1. ``sample_type`` class attribute declares which schema class the bridge handles.
2. ``start(node)`` is called once after rclpy.init(); all publisher creation happens here.
3. ``on_sample(sample)`` is called (potentially from multiple threads) for each new datum.
4. ``stop()`` is idempotent.

ROS 2 imports MUST live inside ``start()`` or later — never at module level.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar


class Bridge(ABC):
    """Base class for all ros2_visualization bridges."""

    #: The schema sample type this bridge handles.  Set on each subclass.
    sample_type: ClassVar[type]

    #: Human-readable name used for logging and registry keys.
    name: ClassVar[str]

    def __init__(self) -> None:
        self._node: Any = None
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self, node: Any) -> None:
        """Initialise publishers using *node*.  Called once after rclpy.init().

        Subclasses must call ``super().start(node)`` first.
        """
        self._node = node
        self._started = True

    def stop(self) -> None:
        """Tear down any background resources.  Idempotent."""
        self._started = False

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    @abstractmethod
    def on_sample(self, sample: Any) -> None:
        """Translate *sample* and publish to ROS 2.

        May be called from any thread; implementations must be thread-safe.
        """

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _check_started(self) -> None:
        if not self._started:
            raise RuntimeError(
                f"{self.__class__.__name__}.on_sample() called before start()"
            )

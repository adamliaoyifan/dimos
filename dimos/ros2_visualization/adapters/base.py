# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Adapter ABC — the only interface a new project needs to implement.

**Zero ROS imports in this file.**

An Adapter subscribes to a project's native data sources (DimOS streams,
ROS bags, simulation API, etc.) and forwards typed schema samples to a
``BridgeRegistry`` instance via ``registry.publish(sample)``.

Adding visualization support for a new project requires:

1. Subclass ``Adapter`` (≈ 150–300 lines).
2. In ``bind(registry)``, subscribe to your data sources and call
   ``registry.publish(SomeSample(...))`` on each datum.
3. Pass the adapter to :func:`~dimos.ros2_visualization.cli.run.main`.

Nothing in ``core/`` or ``bridges/`` will ever import from your adapter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dimos.ros2_visualization.core.registry import BridgeRegistry


class Adapter(ABC):
    """Base class for all source adapters.

    Attributes
    ----------
    name : str
        Human-readable identifier shown in CLI output and log messages.
    """

    name: str = "unnamed_adapter"

    @abstractmethod
    def bind(self, registry: "BridgeRegistry") -> None:
        """Connect to data sources and wire up ``registry.publish()`` callbacks.

        Called once by the framework after ``BridgeRegistry.start()`` returns.
        Must be non-blocking: spawn threads or subscribe to observables here;
        do not block the caller.

        Args:
            registry: The live BridgeRegistry to publish samples into.
        """

    @abstractmethod
    def shutdown(self) -> None:
        """Cancel subscriptions and clean up resources.

        Must be idempotent — may be called even if ``bind()`` was never called.
        """

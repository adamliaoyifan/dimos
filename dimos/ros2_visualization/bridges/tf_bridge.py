# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""TFBridge — wraps TFBroker as a registered Bridge so it participates in
the registry lifecycle (start / stop).

OdomBridge already calls ``tf_broker.update(sample)`` directly; this bridge
exists so the TFBroker can be registered and started alongside other bridges
without requiring the caller to manage it separately.
"""

from __future__ import annotations

from typing import Any

from dimos.ros2_visualization.core.bridge_base import Bridge
from dimos.ros2_visualization.core.frames import FrameIds
from dimos.ros2_visualization.core.schema import OdomSample
from dimos.ros2_visualization.core.tf_broker import TFBroker


class TFBridge(Bridge):
    """Lifecycle wrapper around TFBroker.

    Register this bridge **and** pass the same ``tf_broker`` instance to
    ``OdomBridge`` so both share one broadcaster.
    """

    sample_type = OdomSample
    name = "tf"

    def __init__(self, frames: FrameIds | None = None) -> None:
        super().__init__()
        self.tf_broker = TFBroker(frames)

    def start(self, node: Any) -> None:
        super().start(node)
        self.tf_broker.start(node)

    def on_sample(self, sample: OdomSample) -> None:
        self._check_started()
        self.tf_broker.update(sample)

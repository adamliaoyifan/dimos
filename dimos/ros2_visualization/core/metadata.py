# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Metadata sidecar topic helper.

Every bridge that produces a ``MarkerArray`` also creates a parallel
``std_msgs/String`` topic named ``<marker_topic>/metadata`` carrying a
JSON-encoded mapping of ``{marker_id: {key: value, ...}}``.

Foxglove's custom panel (ships in ``panels/foxglove/dimos_default.json``)
subscribes to this topic and displays the metadata when a marker is clicked.
"""

from __future__ import annotations

import json
from typing import Any


def encode_metadata(id_to_meta: dict[str | int, dict[str, Any]]) -> str:
    """Serialise a ``{marker_id: metadata_dict}`` mapping to JSON string."""
    return json.dumps({str(k): v for k, v in id_to_meta.items()}, default=str)


def decode_metadata(payload: str) -> dict[str, dict[str, Any]]:
    """Deserialise a JSON metadata payload back to a dict."""
    return json.loads(payload)  # type: ignore[no-any-return]


class MetadataSidecar:
    """Thin wrapper that owns a ``std_msgs/String`` publisher on ``<topic>/metadata``."""

    def __init__(self, node: Any, marker_topic: str) -> None:
        from std_msgs.msg import String  # type: ignore[import-untyped]

        from dimos.ros2_visualization.core.qos import QoSProfile, build

        self._pub = node.create_publisher(
            String,
            f"{marker_topic}/metadata",
            build(QoSProfile.RELIABLE),
        )
        self._String = String

    def publish(self, id_to_meta: dict[str | int, dict[str, Any]]) -> None:
        msg = self._String()
        msg.data = encode_metadata(id_to_meta)
        self._pub.publish(msg)

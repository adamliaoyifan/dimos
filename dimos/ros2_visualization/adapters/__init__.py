# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Adapter layer — project-specific shims that feed typed samples into BridgeRegistry.

adapters/base.py is pure Python (zero ROS imports).
Sub-packages (dimos/, rosbag/) may import project-specific libraries.
"""

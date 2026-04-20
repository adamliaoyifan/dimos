# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""ROS 2 bridge implementations — one file per output message type.

All rclpy imports are deferred to Bridge.start() so the package is importable
without a live ROS 2 installation (e.g. in unit tests with mocked rclpy).
"""

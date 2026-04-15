# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pathlib import Path

# Video/Camera constants
VIDEO_WIDTH = 320
VIDEO_HEIGHT = 240
VIDEO_CAMERA_FOV = 45  # MuJoCo default FOV for head_camera (degrees)
DEPTH_CAMERA_FOV = 160

# Simulated Intel RealSense D435i camera constants
D435I_WIDTH = 640
D435I_HEIGHT = 480
D435I_FOV = 42.7  # degrees, derived from fy=614, h=480

# Simulated Intel RealSense D455i camera constants
# FOV derived from default fy=382 at h=480: 2*atan(240/382) ≈ 57 deg vertical
D455I_WIDTH = 640
D455I_HEIGHT = 480
D455I_FOV = 57.0  # degrees (wider FOV than D435i)

# Depth camera range/filtering constants
MAX_RANGE = 3
MIN_RANGE = 0.2
MAX_HEIGHT = 1.2

# Lidar constants
LIDAR_RESOLUTION = 0.05

# Simulation timing constants
VIDEO_FPS = 20
LIDAR_FPS = 2

LAUNCHER_PATH = Path(__file__).parent / "mujoco_process.py"

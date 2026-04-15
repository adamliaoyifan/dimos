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

import logging
import math
import sys
from threading import Thread
import time
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import Field
from reactivex.disposable import Disposable
from reactivex.observable import Observable
import rerun.blueprint as rrb

from dimos.agents.annotation import skill
from dimos.core.core import rpc
from dimos.core.global_config import GlobalConfig
from dimos.core.module import Module, ModuleConfig
from dimos.core.module_coordinator import ModuleCoordinator
from dimos.core.stream import In, Out
from dimos.core.transport import LCMTransport, pSHMTransport
from dimos.spec.perception import Camera, Pointcloud

if TYPE_CHECKING:
    from dimos.core.rpc_client import ModuleProxy
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Transform import Transform
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.CameraInfo import CameraInfo
from dimos.msgs.sensor_msgs.Image import Image, ImageFormat
from dimos.msgs.sensor_msgs.PointCloud2 import PointCloud2
from dimos.robot.unitree.connection import UnitreeWebRTCConnection
from dimos.utils.data import get_data
from dimos.utils.decorators.decorators import simple_mcache
from dimos.utils.testing.replay import TimedSensorReplay, TimedSensorStorage

if sys.version_info < (3, 13):
    from typing_extensions import TypeVar
else:
    from typing import TypeVar

logger = logging.getLogger(__name__)


def _pitch_to_quaternion(pitch: float) -> Quaternion:
    """Convert a pitch angle (radians, positive = tilt downward) to a quaternion.

    Rotation around the Y-axis: qx=0, qy=sin(pitch/2), qz=0, qw=cos(pitch/2).
    """
    half = pitch / 2.0
    return Quaternion(0.0, math.sin(half), 0.0, math.cos(half))


def _build_odom_tf(
    odom: "PoseStamped",
    rs_x: float = 0.15,
    rs_y: float = 0.0,
    rs_z: float = 0.28,
    rs_pitch: float = 0.10,
) -> "list[Transform]":
    """Build the full TF transform list from an odometry message.

    Parameters are the RealSense camera extrinsics in the base_link frame.
    Used by GO2Connection._odom_to_tf (instance method) and standalone utilities.
    """
    camera_link = Transform(
        translation=Vector3(0.3, 0.0, 0.0),
        rotation=Quaternion(0.0, 0.0, 0.0, 1.0),
        frame_id="base_link",
        child_frame_id="camera_link",
        ts=odom.ts,
    )

    camera_optical = Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        frame_id="camera_link",
        child_frame_id="camera_optical",
        ts=odom.ts,
    )

    rs_link = Transform(
        translation=Vector3(rs_x, rs_y, rs_z),
        rotation=_pitch_to_quaternion(rs_pitch),
        frame_id="base_link",
        child_frame_id="realsense_link",
        ts=odom.ts,
    )

    rs_optical = Transform(
        translation=Vector3(0.0, 0.0, 0.0),
        rotation=Quaternion(-0.5, 0.5, -0.5, 0.5),
        frame_id="realsense_link",
        child_frame_id="realsense_optical",
        ts=odom.ts,
    )

    return [
        Transform.from_pose("base_link", odom),
        camera_link,
        camera_optical,
        rs_link,
        rs_optical,
    ]


class ConnectionConfig(ModuleConfig):
    ip: str = Field(default_factory=lambda m: m["g"].robot_ip)
    # RealSense extrinsics (base_link frame, metres / radians).
    # Populated from GlobalConfig which is set by the active vln_config.yaml camera profile.
    realsense_x: float = Field(default_factory=lambda m: m["g"].realsense_x)
    realsense_y: float = Field(default_factory=lambda m: m["g"].realsense_y)
    realsense_z: float = Field(default_factory=lambda m: m["g"].realsense_z)
    realsense_pitch: float = Field(default_factory=lambda m: m["g"].realsense_pitch)
    # RealSense intrinsics: flat 9-element row-major K matrix [fx,0,cx,0,fy,cy,0,0,1].
    # When set, a CameraInfo is published on the realsense_camera_info stream.
    realsense_intrinsic: list[float] | None = Field(
        default_factory=lambda m: m["g"].realsense_intrinsic
    )
    realsense_width: int = Field(default_factory=lambda m: m["g"].realsense_width)
    realsense_height: int = Field(default_factory=lambda m: m["g"].realsense_height)


class Go2ConnectionProtocol(Protocol):
    """Protocol defining the interface for Go2 robot connections."""

    def start(self) -> None: ...
    def stop(self) -> None: ...
    def lidar_stream(self) -> Observable: ...  # type: ignore[type-arg]
    def odom_stream(self) -> Observable: ...  # type: ignore[type-arg]
    def video_stream(self) -> Observable: ...  # type: ignore[type-arg]
    def move(self, twist: Twist, duration: float = 0.0) -> bool: ...
    def standup(self) -> bool: ...
    def liedown(self) -> bool: ...
    def balance_stand(self) -> bool: ...
    def set_obstacle_avoidance(self, enabled: bool = True) -> None: ...
    def publish_request(self, topic: str, data: dict) -> dict: ...  # type: ignore[type-arg]


def _camera_info_static() -> CameraInfo:
    fx, fy, cx, cy = (819.553492, 820.646595, 625.284099, 336.808987)
    width, height = (1280, 720)

    return CameraInfo(
        frame_id="camera_optical",
        height=height,
        width=width,
        distortion_model="plumb_bob",
        D=[0.0, 0.0, 0.0, 0.0, 0.0],
        K=[fx, 0.0, cx, 0.0, fy, cy, 0.0, 0.0, 1.0],
        R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        binning_x=0,
        binning_y=0,
    )


def make_connection(ip: str | None, cfg: GlobalConfig) -> Go2ConnectionProtocol:
    connection_type = cfg.unitree_connection_type

    if ip in ("fake", "mock", "replay") or connection_type == "replay":
        dataset = cfg.replay_dir
        return ReplayConnection(dataset=dataset)
    elif ip == "mujoco" or connection_type == "mujoco":
        from dimos.robot.unitree.mujoco_connection import MujocoConnection

        return MujocoConnection(cfg)
    else:
        assert ip is not None, "IP address must be provided"
        return UnitreeWebRTCConnection(ip)


class ReplayConnection(UnitreeWebRTCConnection):
    # we don't want UnitreeWebRTCConnection to init
    def __init__(  # type: ignore[no-untyped-def]
        self,
        dataset: str = "go2_sf_office",
        **kwargs,
    ) -> None:
        self.dir_name = dataset
        get_data(self.dir_name)
        self.replay_config = {
            "loop": kwargs.get("loop", True),
            "seek": kwargs.get("seek"),
            "duration": kwargs.get("duration"),
        }

    def connect(self) -> None:
        pass

    def start(self) -> None:
        pass

    def standup(self) -> bool:
        return True

    def liedown(self) -> bool:
        return True

    def balance_stand(self) -> bool:
        return True

    def set_obstacle_avoidance(self, enabled: bool = True) -> None:
        pass

    @simple_mcache
    def lidar_stream(self):  # type: ignore[no-untyped-def]
        lidar_store = TimedSensorReplay(f"{self.dir_name}/lidar")  # type: ignore[var-annotated]
        return lidar_store.stream(**self.replay_config)  # type: ignore[arg-type]

    @simple_mcache
    def odom_stream(self):  # type: ignore[no-untyped-def]
        odom_store = TimedSensorReplay(f"{self.dir_name}/odom")  # type: ignore[var-annotated]
        return odom_store.stream(**self.replay_config)  # type: ignore[arg-type]

    # we don't have raw video stream in the data set
    @simple_mcache
    def video_stream(self):  # type: ignore[no-untyped-def]
        # Legacy Unitree recordings can have RGB bytes that were tagged/assumed as BGR.
        # Fix at replay-time by coercing everything to RGB before publishing/logging.
        def _autocast_video(x):  # type: ignore[no-untyped-def]
            # If the old recording tagged it as BGR, relabel to RGB (do NOT channel-swap again).
            if isinstance(x, Image):
                if x.format == ImageFormat.BGR:
                    x.format = ImageFormat.RGB
                if not x.frame_id:
                    x.frame_id = "camera_optical"
                return x

            # Some recordings may store raw arrays or frame wrappers.
            arr = x.to_ndarray(format="rgb24") if hasattr(x, "to_ndarray") else x
            return Image.from_numpy(arr, format=ImageFormat.RGB, frame_id="camera_optical")

        video_store = TimedSensorReplay(f"{self.dir_name}/video", autocast=_autocast_video)  # type: ignore[var-annotated]
        return video_store.stream(**self.replay_config)  # type: ignore[arg-type]

    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        return True

    def publish_request(self, topic: str, data: dict):  # type: ignore[no-untyped-def, type-arg]
        """Fake publish request for testing."""
        return {"status": "ok", "message": "Fake publish"}


_Config = TypeVar("_Config", bound=ConnectionConfig, default=ConnectionConfig)


class GO2Connection(Module[_Config], Camera, Pointcloud):
    default_config = ConnectionConfig  # type: ignore[assignment]

    cmd_vel: In[Twist]
    pointcloud: Out[PointCloud2]
    odom: Out[PoseStamped]
    lidar: Out[PointCloud2]
    color_image: Out[Image]
    depth_image: Out[Image]
    realsense_image: Out[Image]
    realsense_depth: Out[Image]
    camera_info: Out[CameraInfo]
    realsense_camera_info: Out[CameraInfo]

    connection: Go2ConnectionProtocol
    camera_info_static: CameraInfo = _camera_info_static()
    _camera_info_thread: Thread | None = None
    _latest_video_frame: Image | None = None
    _realsense_camera_info_static: CameraInfo | None = None

    @classmethod
    def rerun_views(cls):  # type: ignore[no-untyped-def]
        """Return Rerun view blueprints for GO2 camera visualization."""
        return [
            rrb.Spatial2DView(
                name="Camera",
                origin="world/robot/camera/rgb",
            ),
        ]

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.connection = make_connection(self.config.ip, self.config.g)

        if hasattr(self.connection, "camera_info_static"):
            self.camera_info_static = self.connection.camera_info_static

        self._realsense_camera_info_static = self._make_realsense_camera_info()

    def _make_realsense_camera_info(self) -> CameraInfo | None:
        """Build a CameraInfo for the RealSense camera from ConnectionConfig intrinsics.

        Returns None when no intrinsics are configured (realsense_intrinsic is None).
        """
        k = self.config.realsense_intrinsic
        if not k or len(k) != 9:
            return None

        fx, cx = k[0], k[2]
        fy, cy = k[4], k[5]
        w, h = self.config.realsense_width, self.config.realsense_height

        return CameraInfo(
            frame_id="realsense_optical",
            height=h,
            width=w,
            distortion_model="plumb_bob",
            D=[0.0, 0.0, 0.0, 0.0, 0.0],
            K=list(k),
            R=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            P=[fx, 0.0, cx, 0.0, 0.0, fy, cy, 0.0, 0.0, 0.0, 1.0, 0.0],
        )

    @rpc
    def record(self, recording_name: str) -> None:
        lidar_store: TimedSensorStorage = TimedSensorStorage(f"{recording_name}/lidar")  # type: ignore[type-arg]
        lidar_store.consume_stream(self.connection.lidar_stream())

        odom_store: TimedSensorStorage = TimedSensorStorage(f"{recording_name}/odom")  # type: ignore[type-arg]
        odom_store.consume_stream(self.connection.odom_stream())

        video_store: TimedSensorStorage = TimedSensorStorage(f"{recording_name}/video")  # type: ignore[type-arg]
        video_store.consume_stream(self.connection.video_stream())

    @rpc
    def start(self) -> None:
        super().start()
        if not hasattr(self, "connection"):
            return
        self.connection.start()

        def onimage(image: Image) -> None:
            self.color_image.publish(image)
            self._latest_video_frame = image

        self._disposables.add(self.connection.lidar_stream().subscribe(self.lidar.publish))
        self._disposables.add(self.connection.odom_stream().subscribe(self._publish_tf))
        self._disposables.add(self.connection.video_stream().subscribe(onimage))
        self._disposables.add(Disposable(self.cmd_vel.subscribe(self.move)))

        # Depth stream is only available from MuJoCo simulation
        if hasattr(self.connection, "depth_stream"):
            self._disposables.add(
                self.connection.depth_stream().subscribe(self.depth_image.publish)
            )

        # Simulated D435i streams (MuJoCo only) — published as realsense_image/depth
        # so they appear on the same pSHM channels the real RealSense relay uses.
        if hasattr(self.connection, "d435i_video_stream"):
            self._disposables.add(
                self.connection.d435i_video_stream().subscribe(self.realsense_image.publish)
            )
        if hasattr(self.connection, "d435i_depth_stream"):
            self._disposables.add(
                self.connection.d435i_depth_stream().subscribe(self.realsense_depth.publish)
            )

        self._camera_info_thread = Thread(
            target=self.publish_camera_info,
            daemon=True,
        )
        self._camera_info_thread.start()

        self.standup()
        time.sleep(3)
        self.connection.balance_stand()
        self.connection.set_obstacle_avoidance(self.config.g.obstacle_avoidance)

        # self.record("go2_bigoffice")

    @rpc
    def stop(self) -> None:
        self.liedown()

        if self.connection:
            self.connection.stop()

        if self._camera_info_thread and self._camera_info_thread.is_alive():
            self._camera_info_thread.join(timeout=1.0)

        super().stop()

    def _odom_to_tf(self, odom: PoseStamped) -> list[Transform]:
        # RealSense extrinsics come from ConnectionConfig (set by vln_config.yaml via GlobalConfig).
        return _build_odom_tf(
            odom,
            rs_x=self.config.realsense_x,
            rs_y=self.config.realsense_y,
            rs_z=self.config.realsense_z,
            rs_pitch=self.config.realsense_pitch,
        )

    def _publish_tf(self, msg: PoseStamped) -> None:
        transforms = self._odom_to_tf(msg)
        self.tf.publish(*transforms)
        if self.odom.transport:
            self.odom.publish(msg)

    def publish_camera_info(self) -> None:
        while True:
            self.camera_info.publish(self.camera_info_static)
            if (
                self._realsense_camera_info_static is not None
                and self.realsense_camera_info.transport
            ):
                self.realsense_camera_info.publish(self._realsense_camera_info_static)
            time.sleep(1.0)

    @rpc
    def move(self, twist: Twist, duration: float = 0.0) -> bool:
        """Send movement command to robot."""
        return self.connection.move(twist, duration)

    @rpc
    def standup(self) -> bool:
        """Make the robot stand up."""
        return self.connection.standup()

    @rpc
    def liedown(self) -> bool:
        """Make the robot lie down."""
        return self.connection.liedown()

    @rpc
    def publish_request(self, topic: str, data: dict[str, Any]) -> dict[Any, Any]:
        """Publish a request to the WebRTC connection.
        Args:
            topic: The RTC topic to publish to
            data: The data dictionary to publish
        Returns:
            The result of the publish request
        """
        return self.connection.publish_request(topic, data)

    @skill
    def observe(self) -> Image | None:
        """Returns the latest video frame from the robot camera. Use this skill for any visual world queries.

        This skill provides the current camera view for perception tasks.
        Returns None if no frame has been captured yet.
        """
        return self._latest_video_frame


def deploy(dimos: ModuleCoordinator, ip: str, prefix: str = "") -> "ModuleProxy":
    from dimos.constants import DEFAULT_CAPACITY_COLOR_IMAGE

    connection = dimos.deploy(GO2Connection, ip=ip)

    connection.pointcloud.transport = pSHMTransport(
        f"{prefix}/lidar", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    )
    connection.color_image.transport = pSHMTransport(
        f"{prefix}/image", default_capacity=DEFAULT_CAPACITY_COLOR_IMAGE
    )

    connection.cmd_vel.transport = LCMTransport(f"{prefix}/cmd_vel", Twist)

    connection.camera_info.transport = LCMTransport(f"{prefix}/camera_info", CameraInfo)
    connection.start()

    return connection

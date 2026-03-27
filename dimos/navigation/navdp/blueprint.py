"""NavDP Blueprint — compose NavDP modules with any DimOS robot connection.

Usage
-----

Standalone (with a Go2 robot)::

    from dimos.navigation.navdp import navdp_blueprint
    from dimos.agents.agent import agent
    from dimos.robot.unitree.go2.connection import go2_connection

    my_robot = autoconnect(
        go2_connection(),
        navdp_blueprint,
        agent(),
    ).global_config(n_workers=6)

    my_robot.build()

Or layer on top of an existing DimOS stack::

    from dimos.navigation.navdp import navdp_navigator, navdp_memory, navdp_skills
    from dimos.robot.unitree.go2.blueprints.smart.unitree_go2 import unitree_go2

    my_robot = autoconnect(
        unitree_go2,
        navdp_navigator(navdp_server_url="http://host:8880"),
        navdp_memory(vlm_server_url="http://host:8866"),
        navdp_skills(),
        agent(),
    )
"""

from __future__ import annotations

from dimos.core.blueprints import autoconnect
from dimos.core.transport import pSHMTransport
from dimos.msgs.sensor_msgs import Image

from dimos.navigation.navdp.navigator import navdp_navigator
from dimos.navigation.navdp.memory import navdp_memory
from dimos.navigation.navdp.skills import navdp_skills

# Default NavDP blueprint: navigator + memory + skills
# Expects a robot connection providing: color_image (Out[Image]), odom (Out[PoseStamped])
# Produces: cmd_vel (Out[Twist])
navdp_blueprint = autoconnect(
    navdp_navigator(),
    navdp_memory(),
    navdp_skills(),
).transports({
    # Use shared memory for high-throughput image streaming on same machine
    ("color_image", Image): pSHMTransport("color_image"),
    ("topdown_map", Image): pSHMTransport("topdown_map"),
})

__all__ = ["navdp_blueprint"]

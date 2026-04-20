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

"""Export fused scene to simulation formats (MuJoCo XML, OBJ).

Follows the pattern from dimos.mapping.occupancy.extrude_occupancy for
MuJoCo XML generation.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh  # type: ignore[import-untyped]

from dimos.reconstruction.config import ExportConfig
from dimos.reconstruction.types import FusedScene, MujocoExport, TriangleMesh

logger = logging.getLogger(__name__)


class ExportStage:
    """Export fused scene to MuJoCo XML and mesh files."""

    def __init__(self, config: ExportConfig) -> None:
        self.config = config

    @property
    def name(self) -> str:
        return "export"

    def run(self, input_data: FusedScene) -> MujocoExport:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        mesh_paths: list[Path] = []
        collision_paths: list[Path] = []

        if input_data.mesh is not None:
            # Export visual mesh
            if self.config.export_obj:
                visual_path = output_dir / "visual_mesh.obj"
                export_mesh_to_obj(input_data.mesh, visual_path)
                mesh_paths.append(visual_path)

            # Export collision geometry
            if self.config.export_collision:
                collision_meshes = generate_collision_geometry(
                    input_data.mesh,
                    max_convex_hulls=self.config.max_convex_hulls,
                    target_face_count=self.config.collision_face_count,
                )
                for i, cm in enumerate(collision_meshes):
                    coll_path = output_dir / f"collision_{i:04d}.obj"
                    export_mesh_to_obj(cm, coll_path)
                    collision_paths.append(coll_path)

        # Generate MuJoCo XML
        xml_content = ""
        xml_path = output_dir / "scene.xml"
        if self.config.export_mujoco_xml:
            xml_content = generate_mujoco_xml(
                input_data, self.config, mesh_paths, collision_paths
            )
            xml_path.write_text(xml_content)

        return MujocoExport(
            xml_path=xml_path,
            xml_content=xml_content,
            mesh_paths=mesh_paths,
            collision_mesh_paths=collision_paths,
        )

    def validate_input(self, input_data: FusedScene) -> list[str]:
        errors: list[str] = []
        if input_data.mesh is None and input_data.point_cloud.num_points == 0:
            errors.append("No mesh or point cloud to export")
        return errors


# --- Pure functions ---


def export_mesh_to_obj(mesh: TriangleMesh, output_path: Path) -> Path:
    """Write triangle mesh to OBJ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    tri = trimesh.Trimesh(
        vertices=mesh.vertices,
        faces=mesh.faces,
        vertex_colors=(
            (mesh.vertex_colors * 255).astype(np.uint8)
            if mesh.vertex_colors is not None
            else None
        ),
    )
    tri.export(str(output_path))
    return output_path


def generate_collision_geometry(
    mesh: TriangleMesh,
    max_convex_hulls: int = 32,
    target_face_count: int = 10_000,
) -> list[TriangleMesh]:
    """Approximate convex decomposition for physics collision.

    Uses trimesh's convex decomposition (VHACD when available,
    falls back to single convex hull).
    """
    tri = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)

    try:
        # Try VHACD convex decomposition
        parts = tri.convex_decomposition(maxhulls=max_convex_hulls)
        if not isinstance(parts, list):
            parts = [parts]
    except Exception:
        logger.warning(
            "Convex decomposition failed, falling back to single convex hull"
        )
        parts = [tri.convex_hull]

    result: list[TriangleMesh] = []
    for part in parts:
        # Simplify each collision hull
        if len(part.faces) > target_face_count:
            part = part.simplify_quadric_decimation(target_face_count)

        result.append(
            TriangleMesh(
                vertices=np.asarray(part.vertices, dtype=np.float64),
                faces=np.asarray(part.faces, dtype=np.int64),
            )
        )

    return result


def generate_mujoco_xml(
    scene: FusedScene,
    config: ExportConfig,
    visual_mesh_paths: list[Path],
    collision_mesh_paths: list[Path],
) -> str:
    """Generate MuJoCo MJCF XML referencing exported meshes."""
    root = ET.Element("mujoco", model=config.mujoco_model_name)

    # Compiler settings
    ET.SubElement(root, "compiler", angle="radian", meshdir=".")

    # Visual settings
    visual = ET.SubElement(root, "visual")
    map_elem = ET.SubElement(visual, "map")
    map_elem.set("znear", "0.01")
    map_elem.set("zfar", "100")

    # Assets
    asset = ET.SubElement(root, "asset")

    # Floor texture
    ET.SubElement(
        asset, "texture",
        name="grid", type="2d", builtin="checker",
        rgb1="0.8 0.8 0.8", rgb2="0.6 0.6 0.6",
        width="512", height="512",
    )
    ET.SubElement(asset, "material", name="grid_mat", texture="grid", texrepeat="8 8")

    # Visual meshes
    for i, mp in enumerate(visual_mesh_paths):
        ET.SubElement(asset, "mesh", name=f"visual_{i}", file=mp.name)

    # Collision meshes
    for i, mp in enumerate(collision_mesh_paths):
        ET.SubElement(asset, "mesh", name=f"collision_{i}", file=mp.name)

    # Worldbody
    worldbody = ET.SubElement(root, "worldbody")

    # Floor
    ET.SubElement(
        worldbody, "geom",
        name="floor", type="plane",
        size="50 50 0.1",
        pos=f"0 0 {config.floor_z_offset}",
        material="grid_mat",
        conaffinity="1", condim="3",
    )

    # Light
    ET.SubElement(
        worldbody, "light",
        directional="true",
        diffuse="0.8 0.8 0.8",
        specular="0.2 0.2 0.2",
        pos="0 0 5",
        dir="0 0 -1",
    )

    # Scene body with visual and collision geometry
    scene_body = ET.SubElement(worldbody, "body", name="scene", pos="0 0 0")

    for i, _ in enumerate(visual_mesh_paths):
        ET.SubElement(
            scene_body, "geom",
            name=f"visual_{i}",
            type="mesh", mesh=f"visual_{i}",
            contype="0", conaffinity="0",  # Visual only, no collision
            rgba="0.8 0.8 0.8 1",
        )

    for i, _ in enumerate(collision_mesh_paths):
        ET.SubElement(
            scene_body, "geom",
            name=f"collision_{i}",
            type="mesh", mesh=f"collision_{i}",
            contype="1", conaffinity="1",
            rgba="0.8 0.8 0.8 0",  # Invisible collision geometry
        )

    # Indent for readability
    ET.indent(root)
    return ET.tostring(root, encoding="unicode", xml_declaration=True)

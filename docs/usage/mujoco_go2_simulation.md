# Go2 MuJoCo simulation: config inventory, runbooks, and acceptance

DimOS uses the same MuJoCo backend (`MujocoConnection` + subprocess) for **plain simulation** and for **VLN + agent** stacks. Keep the two workflows separate: they differ by **blueprint** (what modules run) and optional **VLN YAML**, not by a second simulation flag.

| Workflow | Purpose | Typical blueprint |
|----------|---------|-------------------|
| **Local MuJoCo** | Navigation, mapping, teleop, replay-style testing **without** requiring VLN | e.g. `unitree-go2`, `unitree-go2-spatial`, other Go2 stacks with `--simulation` |
| **VLN MuJoCo** | Natural-language goals + VLM/NavDP-style behavior in sim | e.g. `unitree-go2-vln` (and `unitree-go2-vln-local` when present in your tree) with `--simulation` |

Visualization stays on **DimOS paths** only: **WebsocketVis** (map / costmap / paths; default websocket stack on port **7779** with common Go2 blueprints) and **`--viewer`** (`rerun`, `rerun-web`, `rerun-connect`, `foxglove`, `none`). **ROS / RViz are not required** for MuJoCo algorithm or VLN testing in this repository.

---

## 1. `GlobalConfig` — MuJoCo-related fields

[`GlobalConfig`](/dimos/core/global_config.py) holds CLI flags and environment overrides. Precedence: defaults → `.env` / env → blueprint → `dimos [options] run ...`.

Environment variables use the **`DIMOS_` prefix** (e.g. `DIMOS_SIMULATION=true`, `DIMOS_MUJOCO_ROOM=office1`). CLI uses kebab-case (`--mujoco-room`).

| Field | Type / default | Role |
|-------|----------------|------|
| `simulation` | `bool`, default `False` | When `True`, `unitree_connection_type` resolves to **`mujoco`** (see property on `GlobalConfig`). |
| `mujoco_room` | `str \| None`, default `None` | Selects `scene_<name>.xml` from the `mujoco_sim` data directory; if unset, the scene loader uses **`office1`**. |
| `mujoco_room_from_occupancy` | `str \| None` | If set, **scene XML is generated** from an occupancy grid file via `generate_mujoco_scene` instead of loading a static `scene_*.xml`. |
| `mujoco_start_pos` | `str`, default `"-1.0, 1.0"` | Parsed as two numbers; sets the robot root spawn in the MuJoCo process (see `mujoco_start_pos_float`). |
| `mujoco_camera_position` | `str \| None` | Passive **viewer** camera; if `None`, defaults are applied in code (`mujoco_camera_position_float`). |
| `mujoco_steps_per_frame` | `int`, default `7` | Simulation stepping relative to control frames. |
| `mujoco_global_costmap_from_occupancy` | `str \| None` | **Not** used by `load_scene_xml`; used by mapping to seed a global costmap from an occupancy file in sim-like setups (see `dimos/robot/unitree/type/map.py`). |
| `mujoco_global_map_from_pointcloud` | `str \| None` | **Not** scene XML; optional point-cloud-derived map path for mapping modules (see `dimos/mapping/pointclouds/accumulators/general.py`). |

---

## 2. `load_scene_xml` — how the MuJoCo scene is chosen

[`load_scene_xml`](/dimos/simulation/mujoco/model.py) takes the current `GlobalConfig` and returns **XML text** for the scene:

1. **Occupancy-driven scene** — If `mujoco_room_from_occupancy` is set, the path is read as an occupancy grid and **`generate_mujoco_scene`** builds the scene XML.
2. **Static scene asset** — Otherwise `mujoco_room` defaults to **`office1`** when `None`, and the file  
   `scene_<mujoco_room>.xml`  
   is loaded from the **`mujoco_sim`** data directory (`get_data("mujoco_sim")`).

Robot policy and XML wrapping (includes, assets) are handled separately by `get_model_xml` / `load_model`; the important part for “which room” is the above branch.

---

## 3. Multi-room house scene (`scene_house`)

A ready-to-use 8-room house scene (20 m × 14 m) is provided for cross-room VLN testing.

### Room layout

```
      0      6           14     20
 14  +=======+=====+=+====+=======+
     |Master | Bath|L|  Guest BR  |
     |  BR   | room|a|  (8×4)    |
     | (7×4) | (3×4)|u|           |
 10  +=D=====+=D===+D+=====D=====+
     |       |                    |
     |Dining |  OPEN HALLWAY      |Study
     |Room   |  (8 m × 10 m)     |(6×5)
     | (6×5) |                    |
  5  +=D=====+  open plan        +=D====+
     |       |                    |      |
     |Living |  Robot spawns     |Kitchen|
     |Room   |  at (10, 3)       | (6×5)|
     | (6×5) |                    |      |
  0  +=======+====================+======+
D = doorway (1.2 m gap)
```

| Room | Map centroid (x, y) | Notable objects |
|------|---------------------|-----------------|
| `living_room`    | (3.0, 2.5)  | Sofa, coffee table, TV |
| `dining_room`    | (3.0, 7.5)  | Dining table, chairs |
| `kitchen`        | (17.0, 2.5) | Counters, island, fridge, stools |
| `study`          | (17.0, 7.5) | Desk, monitor, bookshelf |
| `master_bedroom` | (3.5, 12.0) | King bed, wardrobe |
| `guest_bedroom`  | (16.0, 12.0)| Bed, dresser |
| `bathroom`       | (8.5, 12.0) | Bathtub, vanity |
| `hallway`        | (10.0, 5.0) | Plants, console table |

### Quick start — house scene

```bash
# 1. Start the VLN-local blueprint in the house scene (robot spawns in hallway):
dimos --simulation --mujoco-room house \ 
      --mujoco-start-pos "10.0, 3.0" \ 
        run unitree-go2-vln-local
# 2. Enable simulation VLM prompt in your VLN config (edit vln_config.yaml):
#      deployment:
#        enabled: true
#        room_layout: "configs/sim/room_layout_house.yaml"

# 3. Send a cross-room VLN goal (second terminal):
dimos agent-send "find the desk chair in the study"
dimos agent-send "go to the kitchen"
dimos agent-send "find the bathtub in the bathroom"
```

### Room layout pre-seeding

The file [`configs/sim/room_layout_house.yaml`](/configs/sim/room_layout_house.yaml) defines all 8 room centroids.  
When `deployment.room_layout` is set in `vln_config.yaml`, `VLNSkillContainer` seeds `SpatialMemory.tag_location` for each room at startup (5 s after module start). This enables:

- `navigate_with_text("kitchen")` → navigates immediately to the pre-seeded centroid.
- `find_object_in_room("fridge in the kitchen")` → jumps to kitchen centroid first, then runs the VLM object search.

To activate for the house scene, set in `dimos/agents/skills/config/vln_config.yaml` (or your custom `VLN_CONFIG`):

```yaml
deployment:
  enabled: true            # simulation VLM prompt prefix
  room_layout: "configs/sim/room_layout_house.yaml"
```

### Switching between scenes

| Goal | Command |
|------|---------|
| Default office (previous behavior) | `dimos --simulation run unitree-go2-vln-local` (no `--mujoco-room`) |
| Empty flat world | `dimos --simulation run unitree-go2-vln-local --mujoco-room empty` |
| Multi-room house | `dimos --simulation run unitree-go2-vln-local --mujoco-room house --mujoco-start-pos "10.0, 3.0"` |

Scenes are **configuration-only** — same blueprint, same codebase. Your `VLN_CONFIG`, Ollama URLs, NavDP settings, and all other env vars carry over unchanged.

---

## 3a. VLN configuration — YAML `deployment` block (simulation vs hardware)

VLN behavior is configured by module config on [`VLNSkillContainer`](/dimos/agents/skills/vln_skill.py) (`VLNConfig`: `vlm_prompt_prefix`, `depth_scale`, `exploration_mode`, NavDP-related fields, etc.). Some deployments also use a **YAML file** (historically `dimos/agents/skills/config/vln_config.yaml`) loaded via `load_vln_config` / `VLN_CONFIG` for agent URLs, NavDP URLs, camera profiles, and a **`deployment`** section:

| YAML area | Meaning |
|-----------|---------|
| `deployment.enabled` | When `true`, treats the run as **simulation-style** for prompt selection. |
| `deployment.vlm_prompt_prefix` | Long prefix prepended to **every VLM query** in simulation-style deployment (explains synthetic / MuJoCo imagery). |
| `deployment.hardware_vlm_prompt_prefix` | Used when **not** in simulation-style deployment (real-camera context). |
| `deployment.room_layout` | Path to a room layout YAML (e.g. `configs/sim/room_layout_house.yaml`). When non-empty, `VLNSkillContainer` seeds `SpatialMemory.tag_location` for each room at startup. Set to `""` to disable (default). |

Other top-level groups in that YAML typically include `vlm`, `agent`, `search`, `navdp`, `escape`, and `blueprint` (exact schema matches your repo’s `load_vln_config` and `vln_config.yaml`).

**Note:** In Python, the overlapping knobs appear as flat fields on `VLNConfig` (e.g. `vlm_prompt_prefix`, `depth_scale: 1.0` for simulation depth in metres). Prefer **`VLN_CONFIG=/path/to/your.yaml`** (when supported) so shared defaults in the repo stay unchanged.

---

## 4. Tier 1 — Runbooks (canonical commands)

Install sim extras as documented in the project (e.g. `uv sync` with the appropriate extras for MuJoCo).

### 4a. Local MuJoCo (no VLN requirement)

```bash
dimos --simulation --viewer rerun run unitree-go2
```

Pick room and spawn explicitly:

```bash
dimos --simulation --mujoco-room office1 --mujoco-start-pos "-1.0, 1.0" --viewer rerun run unitree-go2
```

Use **WebsocketVis** in the browser for map/costmap/path overlays; use **`--viewer rerun`** (or `rerun-web`, `foxglove`) for sensor / debug visualization as usual.

### 4b. VLN MuJoCo (natural-language navigation in sim)

```bash
dimos --simulation --viewer rerun run unitree-go2-vln
```

Optional: set `VLN_CONFIG` to a copy of the example under `configs/sim/` if your tree loads VLN from YAML. Use the same `mujoco_*` flags as in §4a so **scene and spawn** match your language goals (e.g. cross-room tasks).

**Second terminal — agent messages**

```bash
dimos agent-send "explore the space ahead, then stop near the far doorway"
```

Or use the interactive REPL:

```bash
dimos agent-repl
```

Presets: see `configs/sim/README.md` and `bin/mujoco-go2-*.sh`.

---

## 5. Tier 1 — Acceptance checklist

Use this to confirm a sim session is healthy **without ROS/RViz**.

| Step | Local MuJoCo | VLN MuJoCo |
|------|----------------|------------|
| Process | `dimos run` starts; no immediate worker crash; logs under `~/.local/state/dimos/logs/<run-id>/` | Same |
| MuJoCo | Scene matches `mujoco_room` / occupancy setting; robot pose updates | Same |
| Visualize | Websocket map/costmap/path **and/or** Rerun / Foxglove per `--viewer` — **not** RViz | Same |
| Goal | N/A (or your own nav goal) | At least one **natural-language** goal that expects **measurable pose change across the map** (e.g. “go to the other room / far side of the map”) — confirm in websocket and/or viewer |

---

## 6. Tier 2 — `dimos agent-repl`

Interactive REPL wrapping the same MCP tool as `dimos agent-send` (`agent_send`). Requires a blueprint with **`McpServer`** running (e.g. `unitree-go2-vln`). See `docs/usage/cli.md`.

---

## 7. Optional Tier 3 (out of scope here)

A ROS 2 bridge for RViz or external stacks is **not** required for DimOS MuJoCo; treat it as a separate integration if you need interoperability.

# MuJoCo / VLN presets (non-core)

These files are **documentation and copy-paste presets** only. They do not change DimOS core or the MuJoCo subprocess.

## Files

| File | Purpose |
|------|---------|
| `vln.example.yaml` | Example VLN YAML fragments (deployment prompts, pointers to local services). Copy and customize; point `VLN_CONFIG` at your copy when your blueprint supports it. |
| `presets/local-mujoco.env.example` | Suggested `DIMOS_*` variables for **local** Go2 MuJoCo (no VLN). |
| `presets/vln-mujoco.env.example` | Suggested env for **VLN** MuJoCo; set `VLN_CONFIG` to your YAML path. |

## Usage

```bash
# Example: load preset env vars (review and edit paths first)
set -a
source configs/sim/presets/vln-mujoco.env.example
set +a

dimos --simulation --viewer rerun run unitree-go2-vln
```

Or use the thin wrappers in `bin/mujoco-go2-local-sim.sh` and `bin/mujoco-go2-vln-sim.sh`.

Full narrative: [docs/usage/mujoco_go2_simulation.md](/docs/usage/mujoco_go2_simulation.md).

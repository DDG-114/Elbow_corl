# AGENTS.md

This repository is an existing Unitree Go1 + Isaac Lab + LeWM + MPC foothold-cue project. It is not a greenfield repository.

## Project Goal

The project should align its world-model layer with the original `lucas-maes/le-wm` design:

```text
observation frame o_t
    ↓
encoder
    ↓
latent z_t
    ↓ + action a_t
predictor
    ↓
predicted next latent z_{t+1}
```

For this repository, the target architecture is:

```text
Go1 local heightmap / depth-like frame / observation frame
    ↓
LeWM-style encoder
    ↓
latent z_t
    ↓ + MidAction / high-level command
LeWM-style predictor rollout
    ↓
future latent sequence
    ↓
latent cost / auxiliary probes
    ↓
MPC foothold selector / cue injection
    ↓
existing low-level Go1 locomotion policy
```

## Non-Negotiable Rules

- Do not rewrite the whole repository.
- Do not delete working Phase 1 scaffold unless explicitly requested.
- Do not let LeWM directly output 12D joint actions.
- Do not replace the existing low-level Go1 locomotion policy in Phase 1.
- Do not treat `predict_risk()` as the core world-model objective.
- Risk / terrain / state heads must be auxiliary probe heads.
- Core world-model semantics are:
  - `encode`
  - `encode_frame`
  - `predict_next_latent`
  - `rollout_latent`
- Do not pretend that upstream `lucas-maes/le-wm` checkpoints are directly compatible with Go1.
- Do not import Isaac Lab in unit-testable core modules.
- Do not import test fixtures from runtime scripts.
- Do not silently fallback to baseline for unimplemented modes.
- Do not commit data, checkpoints, logs, runs, HDF5 files, videos, or USD artifacts.

## Allowed Dependency Direction

```text
common
  ↓
envs
  ↓
world_model / foothold / mpc
  ↓
controllers
  ↓
scripts
  ↓
eval
```

Do not introduce circular imports.

## World Model Boundary

The world model is responsible for:

```text
observation-frame encoding
latent transition prediction
short-horizon latent rollout
uncertainty / latent planning scores
optional auxiliary probes
```

The world model is not responsible for:

```text
direct joint action control
torque control
low-level gait generation
replacing the locomotion policy
```

## Auxiliary Probe Rule

These are auxiliary probes:

```text
foothold risk head
terrain traversability head
state prediction head
payload sensitivity head
```

They can guide planning, but they must not define the core LeWM architecture.

## Testing Policy

Run lightweight tests first:

```bash
python -m pytest go1_lewm_mpc/tests -q
```

Simulator-dependent checks should be local and optional:

```bash
./isaaclab.sh -p scripts/run_baseline.py --help
./isaaclab.sh -p scripts/run_closed_loop.py --world_model dummy --headless --duration_sec 5
```

If Isaac Lab is unavailable, add or run mock tests instead of forcing simulator execution.

## Codex Response Format

Every task response must end with:

```text
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

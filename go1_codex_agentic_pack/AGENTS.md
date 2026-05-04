# AGENTS.md

This file defines how coding agents should work in this repository.

## Project Goal

This repository implements a Unitree Go1 + Isaac Lab + LEWM + MPC foothold-cue system.

The project should reuse an existing low-level Unitree Go1 locomotion policy whenever possible. The LEWM/MPC layer should provide high-level terrain-aware and foothold-aware cues, not replace the low-level locomotion controller.

The first engineering target is a stable Isaac Lab simulation prototype:

```text
Isaac Lab Go1 rough environment
    ↓
ObsAdapter
    ↓
LEWM / DummyLEWM
    ↓
Foothold candidate generation
    ↓
OSQP / heuristic foothold selector
    ↓
Velocity bias / low-level cue
    ↓
Official Go1 locomotion policy
```

## Non-Negotiable Rules

- Do not directly output 12D joint actions from LEWM.
- Do not rewrite the official Go1 low-level locomotion policy unless explicitly instructed.
- Do not introduce WBC, torque-level NMPC, OCS2, Crocoddyl, or full-body MPC in Phase 1.
- Do not commit datasets, checkpoints, logs, videos, or large binary files.
- Do not make Isaac Lab-specific imports at module import time unless the file is explicitly an Isaac Lab wrapper.
- Keep Isaac Lab-specific code isolated behind wrappers.
- Core inter-module data must use dataclasses from `go1_lewm_mpc/common/types.py`.
- Every implementation task must include tests or a mock test when Isaac Lab is unavailable.
- Do not silently degrade to baseline when a requested mode is not implemented. Raise a clear `NotImplementedError`.
- If simulator access is unavailable, implement graceful fallback and mock tests, then clearly report the limitation.

## Architecture Boundary

Allowed dependency direction:

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

Do not create circular imports.

## Development Style

Work in small PR-sized increments.

Preferred pattern:

```text
read task
inspect repo
modify minimal files
run lightweight tests
summarize changes
stop
```

Do not implement future tasks early.

Each task response must include:

```text
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
```

## Testing Policy

Always run lightweight tests first:

```bash
python -m pytest go1_lewm_mpc/tests -q
```

If only one module is modified, run the relevant focused test first, then the full lightweight suite.

Isaac Lab integration tests should be run locally with:

```bash
./isaaclab.sh -p scripts/run_baseline.py --help
./isaaclab.sh -p scripts/run_baseline.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --headless \
  --duration_sec 5
```

If Isaac Lab is unavailable, tests must still pass with fake data.

## Simulator Dependency Rules

Codex cloud or a generic coding environment may not have:

- Isaac Sim
- Isaac Lab
- NVIDIA GPU drivers
- Unitree assets
- RSL-RL checkpoints

Therefore:

- Import Isaac Lab lazily inside wrapper methods.
- Provide clear error messages for missing simulator dependencies.
- Do not make unit tests require Isaac Lab.
- Keep mock tests independent of Isaac Lab.

## Data and Artifact Rules

Never commit:

```text
data/
datasets/
runs/
logs/
checkpoints/
*.hdf5
*.h5
*.pt
*.pth
*.ckpt
*.onnx
*.mp4
*.avi
*.usd
*.usda
```

If a script creates these files, ensure they are covered by `.gitignore`.

## Phase 1 Success Criteria

Phase 1 is successful when:

- Official Go1 rough baseline can be started.
- Core dataclass interfaces are implemented.
- Fake-data tests pass without Isaac Lab.
- DummyLEWM can produce candidate risk scores.
- Candidate generator and OSQP/fallback selector can choose a foothold.
- Cue injection can produce a safe corrected command.
- A dummy closed-loop smoke test can run without NaNs.
- Real Isaac Lab validation can be performed locally when the simulator is available.

## Research Boundary

The first research question is not:

```text
Can LEWM directly control all Go1 joints?
```

The first research question is:

```text
Can LEWM-based terrain/risk prediction provide useful high-level cues that improve rough-terrain and payload robustness compared with the baseline low-level locomotion policy?
```

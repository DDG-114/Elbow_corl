# Environment Setup

This project targets Unitree Go1 locomotion simulation in Isaac Sim / Isaac Lab.

## Recommended Local Environment

```text
OS: Ubuntu 22.04 recommended
GPU: NVIDIA GPU with working driver
Simulator: Isaac Sim
Framework: Isaac Lab
RL backend: rsl_rl
Python: Isaac Lab managed Python
```

## Expected Isaac Lab Task

The default task is:

```text
Isaac-Velocity-Rough-Unitree-Go1-v0
```

This is used as the low-level Go1 rough-terrain locomotion baseline.

## Installation Notes

Install Isaac Lab following the official Isaac Lab instructions for your Isaac Sim version.

Install RSL-RL support:

```bash
./isaaclab.sh -i rsl_rl
```

## Running Python Scripts

Prefer running project scripts through Isaac Lab:

```bash
./isaaclab.sh -p scripts/run_baseline.py --help
```

Run a short Go1 rough baseline:

```bash
./isaaclab.sh -p scripts/run_baseline.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --headless \
  --duration_sec 5
```

Run a dummy closed-loop smoke test after later tasks are implemented:

```bash
./isaaclab.sh -p scripts/run_closed_loop.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --duration_sec 10 \
  --world_model dummy \
  --use_mpc true \
  --use_cue true \
  --headless
```

## Cloud Codex Limitation

Codex cloud or generic coding sandboxes may not have:

- Isaac Sim
- Isaac Lab
- GPU drivers
- RSL-RL checkpoints
- Unitree assets

Therefore:

- Unit tests must pass without Isaac Lab.
- Isaac Lab imports should be lazy.
- Scripts should fail gracefully when Isaac Lab is unavailable.
- Real simulation verification should be performed locally.

## Local Validation Checklist

Before asking Codex to implement simulator-dependent tasks, check:

```bash
nvidia-smi
./isaaclab.sh -p -c "import isaaclab; print('isaaclab ok')"
./isaaclab.sh -p scripts/run_baseline.py --help
```

After PR-1:

```bash
./isaaclab.sh -p scripts/run_baseline.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 4 \
  --headless \
  --duration_sec 5
```

## Suggested Repository Layout

```text
go1_lewm_mpc/
├── AGENTS.md
├── TASKS.md
├── CODEX_TASK_QUEUE.md
├── docs/
│   └── ENVIRONMENT.md
├── go1_lewm_mpc/
├── configs/
├── scripts/
└── runs/
```

## Debugging Common Issues

### Isaac Lab import error

Use Isaac Lab launcher:

```bash
./isaaclab.sh -p your_script.py
```

Do not use system Python unless Isaac Lab is installed into that environment.

### Task not found

Check that the task name is correct:

```text
Isaac-Velocity-Rough-Unitree-Go1-v0
```

If the task registry has changed, update only the config and wrapper, not the high-level architecture.

### No display / remote server

Use:

```bash
--headless
```

### GPU memory issue

Reduce:

```text
--num_envs
```

Start with 4 or 8 environments for debugging.

## Project-Specific Principle

Simulator integration should be thin.

Most project logic should be testable without Isaac Lab:

```text
ObsPacket
DummyLEWM
Candidate generator
OSQP selector
Cue injection
Metrics
```

Only these files should directly depend on Isaac Lab:

```text
go1_lewm_mpc/envs/go1_env_wrapper.py
go1_lewm_mpc/envs/obs_adapter.py
scripts/run_baseline.py
scripts/collect_dataset.py
scripts/run_closed_loop.py
scripts/eval_closed_loop.py
```

# Environment Notes

## Local Environment

This project is expected to run Isaac Sim / Isaac Lab locally.

Recommended:

```text
Ubuntu 22.04
NVIDIA GPU
Isaac Sim
Isaac Lab
rsl_rl
```

Run scripts through Isaac Lab:

```bash
./isaaclab.sh -p scripts/run_baseline.py --help
```

## Default Go1 Task

```text
Isaac-Velocity-Rough-Unitree-Go1-v0
```

## Important Limitation

Codex cloud or generic coding sandboxes may not have Isaac Lab or GPU access.

Therefore:

```text
unit tests must pass without Isaac Lab
Isaac Lab imports should be lazy
runtime mock mode should exist
simulator validation should be local
```

## Suggested Local Checks

```bash
python -m pytest go1_lewm_mpc/tests -q
./isaaclab.sh -p scripts/run_baseline.py --help
./isaaclab.sh -p scripts/run_baseline.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 4 \
  --headless \
  --duration_sec 5
```

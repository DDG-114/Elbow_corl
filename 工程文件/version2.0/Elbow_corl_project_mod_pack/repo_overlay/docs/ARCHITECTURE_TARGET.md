# Target Architecture

## Phase 1 Existing Scaffold

```text
Go1 Isaac Lab rough env
    ↓
ObsAdapter
    ↓
DummyLEWM / LEWMAdapter
    ↓
Foothold candidate generator
    ↓
OSQP foothold selector
    ↓
Cue injection
    ↓
Go1 low-level locomotion policy
```

## LeWM-Aligned Target

```text
Go1 Isaac Lab rough env
    ↓
ObsAdapter
    ↓
WorldModelInputFrame builder
    ↓
LeWM encoder
    ↓
latent z_t
    ↓ + MidAction sequence
LeWM predictor rollout
    ↓
latent sequence z_{t:t+H}
    ↓
latent planner / auxiliary probes
    ↓
MPC foothold selector
    ↓
Cue injection
    ↓
existing low-level locomotion policy
```

## Module Responsibilities

### `envs`

Simulator wrapping and observation extraction only.

### `world_model`

LeWM-style latent dynamics.

### `foothold`

Candidate generation and phase estimation.

### `mpc`

Optimization / selection using risk and latent rollout cost.

### `controllers`

Low-level command cue injection, command filtering, policy wrapper.

### `eval`

Metrics, ablations, benchmark cases.

## Critical Boundary

LeWM must not directly control Go1 joints.

Correct:

```text
LeWM → latent cost / candidate score → MPC → velocity cue → policy
```

Wrong:

```text
LeWM → 12D action → Go1 joints
```

# LeWM Alignment Notes

## Original LeWM Semantics

The upstream `lucas-maes/le-wm` project is built around a latent world-model idea:

```text
observation o_t
    ↓
encoder
    ↓
latent z_t
    ↓ + action a_t
predictor
    ↓
predicted latent z_{t+1}
```

Training focuses on latent prediction and regularization, rather than directly predicting a task-specific label.

## Required Interpretation in This Repository

For Unitree Go1 + Isaac Lab, we should not treat the world model as a direct foothold-risk predictor.

Instead:

```text
heightmap / local depth-like frame / observation frame
    ↓
encoder
    ↓
latent
    ↓ + MidAction
predictor
    ↓
future latent sequence
    ↓
latent cost / auxiliary probes
    ↓
MPC / cue injection
```

## Core vs Auxiliary

Core LeWM functions:

```text
encode
encode_frame
predict_next_latent
rollout_latent
```

Auxiliary probes:

```text
predict_risk
predict_state
terrain_head
payload_head
stability_head
```

The auxiliary probes can be useful for evaluation and planning, but they should not define the world model itself.

## Key Mismatch to Fix

| Current Direction | Target LeWM-Aligned Direction |
|---|---|
| risk head as core output | latent dynamics as core |
| ObsPacket directly into risk | observation frame into encoder |
| action unclear | MidAction / high-level action vector |
| one-step risk ranking | latent rollout + planning score |
| dummy risk controls planner | dummy latent model + auxiliary risk probe |
| upstream checkpoint assumed compatible | explicit bridge + mock mode first |

## Action Conditioning

Do not use low-level 12D joint action as LeWM action in Phase 1.

Use a high-level action vector:

```text
[vx, vy, yaw_rate,
 dvx, dvy, dyaw,
 selected_leg_onehot(4),
 foothold_delta_xyz]
```

This keeps LeWM connected to planning decisions without taking over low-level locomotion.

## Planning Direction

Short term:

```text
auxiliary risk + OSQP foothold selector
```

Medium term:

```text
latent rollout cost + OSQP selector
```

LeWM-style:

```text
latent CEM planner proposes high-level action sequence
MPC/cue layer executes safe subset
```

# PR-04 Design Note — MidAction

## Purpose

LeWM needs action conditioning. For Go1, action must not be low-level 12D joint action in Phase 1.

## Proposed Type

```python
@dataclass
class MidAction:
    t: float
    cmd_vel: np.ndarray
    velocity_bias: np.ndarray
    selected_leg_id: int | None
    foothold_delta_b: np.ndarray | None
```

## Suggested Vector

```text
[vx, vy, yaw_rate,
 dvx, dvy, dyaw,
 selected_leg_onehot(4),
 foothold_delta_xyz]
```

Total dim:

```text
13
```

## Why

This lets LeWM predict latent dynamics under high-level planning decisions, while the existing locomotion policy still handles gait and joint control.

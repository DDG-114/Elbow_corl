# PR-03 Design Note — WorldModelBase Refactor

## Current Risk

If `WorldModelBase` only exposes `predict_risk()`, the project becomes a task-specific risk estimator rather than a LeWM-style world model.

## Target Interface

```python
class WorldModelBase(ABC):
    def encode(self, obs: ObsPacket) -> LatentPacket:
        ...

    def encode_frame(self, frame: WorldModelInputFrame) -> LatentPacket:
        ...

    def predict_next_latent(self, latent: LatentPacket, action: np.ndarray) -> LatentPacket:
        ...

    def rollout_latent(self, obs: ObsPacket, action_sequence: np.ndarray, dt: float) -> list[LatentPacket]:
        ...

    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        ...

    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        ...
```

## Semantics

Core:

```text
encode
encode_frame
predict_next_latent
rollout_latent
```

Auxiliary:

```text
predict_risk
predict_state
```

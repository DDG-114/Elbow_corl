# PR-02 Design Note — WorldModelInputFrame

## Purpose

LeWM expects an observation frame-like input. Go1 currently provides proprio and height_scan. To align the project with LeWM, create a frame abstraction without forcing RGB input immediately.

## Proposed Type

```python
@dataclass
class WorldModelInputFrame:
    t: float
    frame: np.ndarray          # [C, H, W]
    frame_type: str            # "heightmap", "depth", "rgb"
    action_context: np.ndarray
    metadata: dict
```

## Conversion

```python
obs.height_scan None        -> zeros [1,64,64]
obs.height_scan [Nh]        -> interpolate/reshape [1,64,64]
obs.height_scan [H,W]       -> resize [1,64,64]
```

## Why This Matters

This prevents the world model from becoming a direct proprio-to-risk predictor. It creates a LeWM-compatible observation-frame boundary.

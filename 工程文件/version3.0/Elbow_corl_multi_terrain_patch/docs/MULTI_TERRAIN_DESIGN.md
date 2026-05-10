# Multi-Terrain Locomotion Design

## Goal

Extend the existing Go1 LEWM-MPC scaffold to support locomotion over:

1. flat ground,
2. narrow beam / single-plank bridge,
3. stepping stones / plum-blossom piles.

## Main design choice

The existing official Go1 rough-terrain policy remains the baseline. This patch
adds terrain-aware planning and evaluation modules without replacing the official
low-level policy.

The long-term target is a terrain-conditioned or foothold-conditioned policy:

```text
robot proprioception
+ velocity command
+ terrain features
+ support map / terrain embedding
+ selected foothold hint
-> 12D joint target action
-> PD control
```

## Why velocity bias is not enough

For flat rough terrain, soft velocity correction can work. For narrow beams and
stepping stones, the controller must reason about exact support regions. A
selected foothold should eventually influence either:

1. the low-level policy observation, or
2. a real foot-placement / whole-body controller.

This patch prepares the first route.

## New components

- `common/terrain_types.py`: `TerrainId`, `TerrainContext`, terrain feature vector.
- `terrains/`: mock-testable flat, beam, stepping-stone, and mixed terrain generators.
- `foothold/terrain_aware_generator.py`: terrain-aware candidate generation.
- `mpc/terrain_cost_terms.py`: support, edge, centerline, and stone-center costs.
- `controllers/foothold_conditioned_cue.py`: richer cue for future policy inputs.
- `eval/terrain_metrics.py`: beam/stones-specific evaluation metrics.

## Terrain-specific objectives

### Flat

- velocity tracking,
- stable torso,
- low energy,
- smooth actions.

### Beam

- keep body near beam centerline,
- align yaw with beam direction,
- keep all contacts inside support map,
- reduce lateral drift and roll.

### Stepping stones

- place swing feet inside stone support regions,
- prefer stone centers over edges,
- maintain sufficient foot clearance,
- progress to next available support.

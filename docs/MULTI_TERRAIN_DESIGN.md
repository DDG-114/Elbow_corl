# Multi-Terrain Locomotion Design

## Goal

Extend the Go1 LeWM-MPC scaffold to support flat ground, narrow beams, and
stepping-stone terrain while preserving the existing low-level Go1 locomotion
policy.

## Design

The official Go1 rough-terrain policy remains the baseline. Terrain-aware
modules add support maps, candidate filtering, terrain-specific costs, richer
future-policy cues, and evaluation metrics around the existing controller.

The world model remains responsible for LeWM-style encoding and latent
prediction. Terrain risk, traversability, and state heads are auxiliary probes;
they do not replace the core `encode`, `encode_frame`, `predict_next_latent`,
and `rollout_latent` semantics.

## Terrain Objectives

- Flat: velocity tracking, stable torso, low energy, smooth actions.
- Beam: stay near the beam centerline, keep contacts inside support, reduce
  lateral drift, and preserve heading alignment.
- Stepping stones: select support-region footholds, prefer stone centers, keep
  foot clearance, and progress to reachable supports.

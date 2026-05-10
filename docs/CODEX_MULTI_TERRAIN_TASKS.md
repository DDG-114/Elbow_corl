# Codex Multi-Terrain Task Queue

This document records the archived `version3.0` multi-terrain patch as an
incremental queue for this repository. The root-level `CODEX_TASK_QUEUE.md`
remains the active source of truth for LeWM alignment and project status.

## Integration Order

1. Add `TerrainId`, `TerrainContext`, and support-map utilities.
2. Add flat, beam, stepping-stone, and mixed mock terrain generators.
3. Add terrain-aware foothold candidate generation while preserving the
   existing nominal generator.
4. Add terrain-specific foothold-selection costs.
5. Add `TerrainAwareFootholdSelector` without removing `OSQPFootholdSelector`.
6. Add richer future-policy cue and policy-observation builders without
   changing the official Go1 policy input shape.
7. Add terrain-specific metrics and benchmark configs.
8. Add debug scripts, mock evaluation, and documentation.

## Guardrails

- Do not rewrite the repository.
- Do not replace the official Go1 policy.
- Do not import Isaac Lab in unit-testable core modules.
- Do not make risk heads the core world-model objective.
- Keep terrain work additive around the current LeWM/MidAction interfaces.

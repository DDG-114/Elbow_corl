# Codex Multi-Terrain Task Queue

Use these tasks in order. Do not ask Codex to rewrite the whole repository.

## Global instruction

```text
Read AGENTS.md and README.md first.
Respect dependency direction:
common -> envs -> world_model / foothold / mpc -> controllers -> scripts -> eval.
Do not import Isaac Lab in unit-testable core modules.
Do not replace the official Go1 policy in this task.
Every response must end with:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

## Task 1: Terrain types and support map

```text
Add multi-terrain data types and support-map utilities.

Files to create:
- go1_lewm_mpc/common/terrain_types.py
- go1_lewm_mpc/terrains/__init__.py
- go1_lewm_mpc/terrains/support_map.py
- go1_lewm_mpc/tests/test_terrain_types.py
- go1_lewm_mpc/tests/test_support_map.py

Files to modify:
- go1_lewm_mpc/common/types.py

Requirements:
- Add TerrainId and TerrainContext.
- Add optional terrain_context to ObsPacket.
- Implement world_xy_to_grid, query_support_value, batch_query_support.
- Keep everything NumPy-only.
- Do not import Isaac Lab.
- Existing tests must still pass.
```

## Task 2: Terrain generators

```text
Add mock-testable terrain generators for flat, beam, and stepping stones.

Files to create:
- go1_lewm_mpc/terrains/base.py
- go1_lewm_mpc/terrains/flat.py
- go1_lewm_mpc/terrains/beam.py
- go1_lewm_mpc/terrains/stepping_stones.py
- go1_lewm_mpc/terrains/mixed.py
- go1_lewm_mpc/terrains/registry.py
- configs/terrain/flat.yaml
- configs/terrain/beam.yaml
- configs/terrain/stepping_stones.yaml
- configs/terrain/mixed.yaml

Requirements:
- Flat support_map must be all ones.
- Beam support_map must mark only beam surface as safe.
- Stepping-stone support_map must mark only circular stone areas as safe.
- No Isaac Lab imports.
```

## Task 3: Terrain-aware foothold generator

```text
Add terrain-aware foothold candidate generation while preserving the existing generator.

Files to create:
- go1_lewm_mpc/foothold/reachability.py
- go1_lewm_mpc/foothold/foothold_utils.py
- go1_lewm_mpc/foothold/terrain_aware_generator.py

Requirements:
- For flat terrain, keep nominal candidates.
- For beam terrain, filter candidates outside support_map.
- For stepping stones, sample candidates around reachable stone centers.
- If no safe candidate exists, return a conservative fallback.
```

## Task 4: Terrain cost terms

```text
Add terrain-aware cost terms for foothold selection.

Files to create:
- go1_lewm_mpc/mpc/terrain_cost_terms.py

Requirements:
- Implement support_violation_cost.
- Implement beam_centerline_cost.
- Implement beam_edge_margin_cost.
- Implement stone_center_cost.
- Implement terrain_total_cost.
```

## Task 5: Selector wrapper

```text
Add TerrainAwareFootholdSelector without deleting OSQPFootholdSelector.

Files to create:
- go1_lewm_mpc/mpc/terrain_aware_selector.py

Requirements:
- Accept obs, swing_leg_id, candidates_b, risk, optional latent_cost.
- Add terrain-specific cost.
- Return a plan-like object with selected_foothold_b and debug info.
```

## Task 6: Foothold-conditioned cue

```text
Add richer cue objects for future low-level policies.

Files to create:
- go1_lewm_mpc/controllers/foothold_conditioned_cue.py
- go1_lewm_mpc/controllers/policy_obs_builder.py
- go1_lewm_mpc/controllers/safety_filter.py

Requirements:
- Do not change official policy input shape.
- Build a future policy observation vector separately.
```

## Task 7: Evaluation metrics

```text
Add terrain-specific evaluation metrics.

Files to create:
- go1_lewm_mpc/eval/terrain_metrics.py
- configs/eval/beam_benchmark.yaml
- configs/eval/stones_benchmark.yaml
- configs/eval/mixed_benchmark.yaml
```

## Task 8: Debug scripts and docs

```text
Add debug scripts and documentation.

Files to create:
- scripts/debug_terrain_generator.py
- scripts/debug_foothold_candidates.py
- scripts/eval_multi_terrain.py
- docs/MULTI_TERRAIN_DESIGN.md
- docs/TERRAIN_EVAL_PROTOCOL.md
```

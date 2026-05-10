# Elbow_corl Multi-Terrain Patch Package

This package contains an additive engineering scaffold for extending the existing `DDG-114/Elbow_corl` project toward multi-terrain locomotion on:

1. flat ground,
2. narrow beam / single-plank bridge,
3. stepping stones / plum-blossom piles.

## Important integration notes

This is **not** a full copy of the original repository. It is a patch-style file set intended to be copied into the repository root and then refined by Codex.

The design intentionally follows these rules:

- Do not rewrite the whole repository.
- Do not delete the existing Phase 1 scaffold.
- Do not import Isaac Lab in unit-testable core modules.
- Keep the official Go1 low-level policy as the baseline.
- Add terrain-aware support maps, candidate generation, cost terms, cue objects, metrics, configs, debug scripts, and documentation.

## Suggested integration command

From the root of the original repository:

```bash
cp -r /path/to/Elbow_corl_multi_terrain_patch/* .
```

Then run tests:

```bash
pytest go1_lewm_mpc/tests -q
python -m compileall go1_lewm_mpc scripts
```

## Recommended Codex order

1. Add and test terrain types + support map.
2. Add flat / beam / stepping-stone terrain generators.
3. Add terrain-aware foothold candidate generator.
4. Add terrain-specific cost terms.
5. Add terrain-aware selector wrapper.
6. Add foothold-conditioned cue and policy observation builder.
7. Add terrain metrics and benchmark configs.
8. Add debug scripts and documentation.

See `docs/CODEX_MULTI_TERRAIN_TASKS.md` for copy-ready prompts.

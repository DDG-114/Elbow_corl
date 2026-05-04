# CODEX_PROMPTS.md

This file contains copy-paste prompts for agentic coding.

Use one prompt per Codex task. Do not ask Codex to implement the whole project in one pass.

---

## Prompt 001 — Project Skeleton

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 001 only.

Goal:
Create the project skeleton, dataclasses, constants, math utilities, config templates, README update, .gitignore, and tests.

Do not implement:
- Isaac Lab integration
- Go1EnvWrapper
- ObsAdapter
- Dataset collector
- LEWM
- MPC
- Cue injection
- Closed-loop runner

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests -q
2. Summarize:
   - Files changed
   - Tests run
   - Assumptions
   - Limitations
   - Next recommended task
3. Stop.
```

---

## Prompt 002 — Baseline Wrapper

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 002 only.

Goal:
Implement a minimal Isaac Lab Go1 baseline wrapper and scripts/run_baseline.py.

Requirements:
- run_baseline.py --help must work without Isaac Lab.
- If Isaac Lab is missing, print a clear actionable message.
- Unit tests must not require Isaac Lab.
- Keep Isaac Lab imports lazy inside wrapper methods.

Do not implement:
- ObsAdapter
- Dataset collector
- LEWM
- MPC
- Cue injection
- Closed-loop runner

After editing:
1. Run: python scripts/run_baseline.py --help
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize changed files, tests, assumptions, limitations, next task.
4. Stop.
```

---

## Prompt 003 — ObsAdapter

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 003 only.

Goal:
Implement ObsAdapter and fake-data tests.

Requirements:
- Implement go1_lewm_mpc/envs/obs_adapter.py.
- Add or update go1_lewm_mpc/tests/fixtures.py.
- Add test_obs_adapter.py.
- Adapter must support fake raw_obs dicts.
- Missing optional fields should fallback with warnings.
- Invalid required shapes should raise ValueError.

Do not implement:
- Dataset collector
- LEWM
- MPC
- Cue injection
- Closed-loop runner

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_obs_adapter.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

---

## Prompt 004 — Dataset Collector

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 004 only.

Goal:
Implement HDF5 dataset schema, writer, replay loader, and scripts/collect_dataset.py.

Requirements:
- Fake data HDF5 write/read tests must pass.
- Do not commit generated HDF5 files.
- Collector should work with Go1EnvWrapper and ObsAdapter when Isaac Lab is available.
- If Isaac Lab is unavailable, script should fail gracefully.

Do not implement:
- LEWM
- MPC
- Candidate generator
- Cue injection
- Closed-loop runner

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_dataset_schema.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

---

## Prompt 005 — DummyLEWM

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 005 only.

Goal:
Implement WorldModelBase and DummyLEWM.

Requirements:
- No torch dependency.
- predict_risk(obs, query_points_b) returns finite shape [K].
- Higher payload should make risk more conservative.
- Far/reachable/z/roughness penalties should be deterministic and testable.

Do not implement:
- Real LEWM
- Candidate generator
- OSQP selector
- Cue injection
- Closed-loop runner

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_dummy_lewm.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

---

## Prompt 006 — Candidate Generator

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 006 only.

Goal:
Implement PhaseEstimator and FootholdCandidateGenerator.

Requirements:
- Generate candidates with shape [K, 3].
- Respect max step x/y/z.
- Forward velocity shifts candidates forward.
- Side velocity shifts candidates sideways.
- Cover all four legs in tests.
- Fall back to fixed trot sequence if contact phase is unavailable.

Do not implement:
- OSQP selector
- Cue injection
- Closed-loop runner

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_candidate_generator.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

---

## Prompt 007 — OSQP Foothold Selector

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 007 only.

Goal:
Implement OSQPFootholdSelector and fallback heuristic selector.

Requirements:
- Avoid high-risk candidates.
- Select near nominal candidate when risks are equal.
- Record solver status and solve time.
- Fallback to argmin total_score if OSQP fails or is unavailable.
- Do not use mixed-integer optimization.

Do not implement:
- Cue injection
- Closed-loop runner
- Evaluation scripts

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_osqp_foothold.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

---

## Prompt 008 — Cue Injection

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 008 only.

Goal:
Convert MpcPlanPacket into LowLevelCue using velocity bias.

Requirements:
- Implement command filtering.
- Implement max bias clipping.
- Implement fake low-level policy wrapper tests.
- Cue disabled mode must match baseline behavior.
- No NaNs.

Do not implement:
- Closed-loop runner
- Evaluation scripts
- Real LEWM

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_cue_injection.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

---

## Prompt 009 — Closed-loop Runner

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 009 only.

Goal:
Implement scripts/run_closed_loop.py and fake closed-loop smoke test.

Requirements:
- Support baseline, no-cue, and cue modes.
- Support dummy world model.
- Detect NaNs and stop safely.
- Log selected foothold, risk, velocity bias, and basic stability info when available.
- Fake smoke test must not require Isaac Lab.

Do not implement:
- Full evaluation benchmark
- Real LEWM
- Ablation table

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

---

## Prompt 010 — Evaluation

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 010 only.

Goal:
Implement repeatable evaluation scripts and metrics.

Requirements:
- Output metrics.csv, summary.json, and config.yaml.
- Separate baseline and cue results.
- Do not fake unavailable metrics. Use NaN with explanation.
- Include git commit hash if available, otherwise "unknown".

Do not implement:
- Real LEWM adapter
- New control algorithm

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_metrics.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

---

## Prompt 011 — Real LEWM Adapter

```text
Read AGENTS.md, TASKS.md, and CODEX_TASK_QUEUE.md.

Implement Task 011 only.

Goal:
Implement LEWMAdapter behind the existing WorldModelBase interface.

Requirements:
- Missing checkpoint should produce a clear error.
- CPU fallback should work with warning.
- predict_risk returns finite [K].
- Closed-loop runner should not need changes to switch from DummyLEWM to LEWMAdapter.

Do not implement:
- New low-level controller
- WBC
- Torque-level MPC
- Direct 12D action output from LEWM

After editing:
1. Run: python -m pytest go1_lewm_mpc/tests/test_lewm_adapter_mock.py -q
2. Run: python -m pytest go1_lewm_mpc/tests -q
3. Summarize and stop.
```

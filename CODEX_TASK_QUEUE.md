# CODEX_TASK_QUEUE.md

This is the agentic coding task queue for the existing `Elbow_corl` repository.

Use `TASKS.md` as the original project plan, but follow this file for incremental LeWM-alignment refactoring.

## Current Status Assumption

The repository already contains most Phase 1 scaffold:

```text
common/types.py
envs/go1_env_wrapper.py
envs/obs_adapter.py
data/HDF5 writer
world_model/base.py
world_model/dummy_lewm.py
world_model/lewm_adapter.py
foothold/candidate_generator.py
mpc/osqp_foothold.py
controllers/cue_injection.py
scripts/run_baseline.py
scripts/run_closed_loop.py
scripts/eval_closed_loop.py
tests/
```

Therefore, do not rebuild the project from scratch.

## Legacy Phase 1 Status

The original task queue in `TASKS.md` has already been implemented far enough
for the current LeWM-alignment work. Treat the existing code as the baseline
for incremental refactoring:

```text
Task 001-008: DONE
Task 009-011: PARTIAL
Task 012: TODO
New LeWM-alignment PRs: TODO
```

Do not re-run the old greenfield queue unless explicitly requested.

---

## PR-00 — Agentic Docs Normalization

Status: DONE

Goal:
Normalize repository-level agentic documentation and task status.

Allowed files:

```text
AGENTS.md
CODEX_TASK_QUEUE.md
CODEX_PROMPTS.md
README.md
.gitignore
```

Do not modify Python runtime code.

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_types.py -q
python -c "from go1_lewm_mpc.common.types import ObsPacket; print('ok')"
```

---

## PR-01 — Runtime Mock Cleanup

Status: DONE

Goal:
Remove runtime imports from `go1_lewm_mpc.tests.*`.

Allowed files:

```text
go1_lewm_mpc/mock/__init__.py
go1_lewm_mpc/mock/fake_isaac_env.py
go1_lewm_mpc/tests/fixtures.py
scripts/eval_closed_loop.py
go1_lewm_mpc/tests/test_closed_loop_smoke.py
go1_lewm_mpc/tests/test_metrics.py
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04
```

---

## PR-02 — WorldModelInputFrame Contract

Status: DONE

Goal:
Add a LeWM-style observation-frame contract.

Allowed files:

```text
go1_lewm_mpc/common/types.py
go1_lewm_mpc/world_model/input_frame.py
go1_lewm_mpc/tests/test_world_model_input_frame.py
go1_lewm_mpc/tests/fixtures.py
configs/env/go1_lewm_rough.yaml
```

Required type:

```python
@dataclass
class WorldModelInputFrame:
    t: float
    frame: np.ndarray          # [C, H, W]
    frame_type: str            # "heightmap", "depth", "rgb"
    action_context: np.ndarray
    metadata: dict
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_world_model_input_frame.py -q
```

---

## PR-03 — LeWM Semantic Interface Refactor

Status: DONE

Goal:
Refactor `WorldModelBase` toward LeWM-style latent dynamics.

Core methods:

```text
encode
encode_frame
predict_next_latent
rollout_latent
```

Auxiliary probe methods:

```text
predict_risk
predict_state
```

Allowed files:

```text
go1_lewm_mpc/world_model/base.py
go1_lewm_mpc/world_model/dummy_lewm.py
go1_lewm_mpc/world_model/lewm_adapter.py
go1_lewm_mpc/tests/test_lewm_semantic_interface.py
go1_lewm_mpc/tests/test_dummy_lewm.py
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_lewm_semantic_interface.py -q
python -m pytest go1_lewm_mpc/tests/test_dummy_lewm.py -q
```

---

## PR-04 — MidAction Contract

Status: DONE

Goal:
Define high-level action conditioning for LeWM predictor.

Allowed files:

```text
go1_lewm_mpc/common/types.py
go1_lewm_mpc/world_model/action_adapter.py
go1_lewm_mpc/tests/test_action_adapter.py
```

Required type:

```python
@dataclass
class MidAction:
    t: float
    cmd_vel: np.ndarray
    velocity_bias: np.ndarray
    selected_leg_id: int | None
    foothold_delta_b: np.ndarray | None
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_action_adapter.py -q
```

---

## PR-05 — World Model Factory

Status: DONE

Goal:
Add a single world-model factory.

Allowed files:

```text
go1_lewm_mpc/world_model/factory.py
go1_lewm_mpc/world_model/lewm_adapter.py
scripts/run_closed_loop.py
scripts/eval_closed_loop.py
go1_lewm_mpc/tests/test_world_model_factory.py
```

Required backends:

```text
dummy
local_lewm
upstream_lewm_mock
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_world_model_factory.py -q
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q
```

---

## PR-06 — Upstream LeWM Bridge Skeleton

Status: DONE

Goal:
Add a boundary layer for `lucas-maes/le-wm`, with mock mode first.

Allowed files:

```text
go1_lewm_mpc/world_model/upstream_lewm_bridge.py
go1_lewm_mpc/world_model/factory.py
configs/lewm/upstream_lewm.yaml
go1_lewm_mpc/tests/test_upstream_lewm_bridge.py
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_upstream_lewm_bridge.py -q
```

---

## PR-07 — LeWM Sequence Dataset Schema

Status: DONE

Goal:
Add LeWM-style training sequences:

```text
frame o_t
action a_t
next_frame o_{t+1}
optional probe labels
```

Allowed files:

```text
go1_lewm_mpc/data/lewm_sequence_dataset.py
go1_lewm_mpc/data/dataset_schema.py
go1_lewm_mpc/data/hdf5_writer.py
go1_lewm_mpc/tests/test_lewm_sequence_dataset.py
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_lewm_sequence_dataset.py -q
```

---

## PR-08 — LeWM Loss and Training Dry Run

Status: DONE

Goal:
Add LeWM-style training structure:

```text
latent prediction loss + lambda_sigreg * SIGReg
```

Allowed files:

```text
go1_lewm_mpc/world_model/lewm_loss.py
go1_lewm_mpc/world_model/simple_lewm_backbone.py
scripts/train_lewm.py
configs/lewm/train_lewm.yaml
go1_lewm_mpc/tests/test_lewm_loss.py
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_lewm_loss.py -q
python scripts/train_lewm.py --config configs/lewm/train_lewm.yaml --dry_run
```

---

## PR-09 — Latent CEM Planner Prototype

Status: DONE

Goal:
Add latent rollout + CEM action sequence search.

Allowed files:

```text
go1_lewm_mpc/world_model/latent_planner.py
go1_lewm_mpc/tests/test_latent_planner.py
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_latent_planner.py -q
```

---

## PR-10 — MPC Cost Uses Latent Rollout

Status: DONE

Goal:
Allow OSQP/selector cost to consume latent rollout cost, not just auxiliary risk.

Allowed files:

```text
go1_lewm_mpc/mpc/osqp_foothold.py
go1_lewm_mpc/mpc/cost_terms.py
scripts/run_closed_loop.py
go1_lewm_mpc/tests/test_osqp_foothold.py
go1_lewm_mpc/tests/test_closed_loop_smoke.py
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_osqp_foothold.py -q
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q
```

---

## PR-11 — Evaluation and Ablation Alignment

Status: DONE

Goal:
Add ablations that distinguish auxiliary risk vs latent rollout cost.

Modes:

```text
baseline
heuristic_only
dummy_risk
local_lewm_aux_risk
local_lewm_latent_cost
upstream_lewm_mock_latent_cost
lewm_no_payload
lewm_no_heightmap
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_metrics.py -q
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04
```

---

## PR-12 — Payload Randomization Bridge

Status: DONE

Goal:
Make payload mass/com propagate through fake env, ObsPacket, dataset and evaluation.

Allowed files:

```text
go1_lewm_mpc/envs/payload_randomization.py
go1_lewm_mpc/envs/go1_env_wrapper.py
go1_lewm_mpc/envs/obs_adapter.py
scripts/collect_dataset.py
scripts/eval_closed_loop.py
go1_lewm_mpc/tests/test_payload_randomization.py
```

Acceptance:

```bash
python -m pytest go1_lewm_mpc/tests/test_payload_randomization.py -q
```

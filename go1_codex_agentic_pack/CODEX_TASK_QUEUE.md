# CODEX_TASK_QUEUE.md

This file is the executable task queue for agentic coding.

Use `TASKS.md` as the master design document. Use this file to execute small, PR-sized tasks.

## Current Phase

Phase 1: Build a mock-testable scaffold and baseline wrapper.

## General Execution Rule

For each task:

1. Read `AGENTS.md`.
2. Read the relevant section of `TASKS.md`.
3. Implement only the current task.
4. Run the requested tests.
5. Summarize changed files, tests, assumptions, limitations, and next task.
6. Stop.

Do not implement later tasks early.

---

## Task 001 — PR-0 Project Skeleton

Status: TODO

### Goal

Create the minimal Python package, configuration files, dataclass-based interfaces, constants, and tests.

### Allowed Files

```text
pyproject.toml
requirements.txt
README.md
.gitignore
configs/env/go1_lewm_rough.yaml
configs/mpc/osqp_foothold.yaml
configs/lewm/dummy_lewm.yaml
go1_lewm_mpc/__init__.py
go1_lewm_mpc/common/__init__.py
go1_lewm_mpc/common/types.py
go1_lewm_mpc/common/constants.py
go1_lewm_mpc/common/math_utils.py
go1_lewm_mpc/tests/__init__.py
go1_lewm_mpc/tests/test_types.py
```

### Requirements

- Define:
  - `ObsPacket`
  - `LatentPacket`
  - `FootholdCandidatePacket`
  - `MpcPlanPacket`
  - `LowLevelCue`
- Define:
  - `FOOT_ORDER = ["FL", "FR", "RL", "RR"]`
  - `N_JOINTS = 12`
  - `N_FEET = 4`
- Add shape validation helper functions in `math_utils.py`.
- Add `.gitignore` rules for data, logs, runs, checkpoints, and simulator artifacts.
- Add README section explaining the Phase 1 architecture.

### Do Not

- Do not import Isaac Lab.
- Do not implement environment wrappers.
- Do not implement LEWM.
- Do not implement MPC.
- Do not implement closed-loop runner.

### Acceptance

```bash
python -m pytest go1_lewm_mpc/tests -q
```

must pass.

`python -c "from go1_lewm_mpc.common.types import ObsPacket"` must work.

---

## Task 002 — PR-1 Baseline Go1 Wrapper

Status: TODO

### Goal

Implement a minimal Isaac Lab Go1 baseline wrapper and runner.

### Allowed Files

```text
go1_lewm_mpc/envs/__init__.py
go1_lewm_mpc/envs/go1_env_wrapper.py
scripts/run_baseline.py
go1_lewm_mpc/tests/test_go1_env_wrapper_mock.py
README.md
```

### Requirements

Implement:

```python
class Go1EnvWrapper:
    def __init__(self, task_name: str, num_envs: int, headless: bool):
        ...

    def reset(self):
        ...

    def step(self, action=None):
        ...

    def get_raw_obs(self):
        ...

    def close(self):
        ...
```

`run_baseline.py` must support:

```bash
python scripts/run_baseline.py --help
python scripts/run_baseline.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --headless \
  --duration_sec 5
```

### Do Not

- Do not implement ObsAdapter.
- Do not implement dataset collection.
- Do not implement LEWM/MPC.
- Do not require Isaac Lab for unit tests.

### Acceptance

- `python scripts/run_baseline.py --help` works.
- If Isaac Lab is missing, the error message is clear and actionable.
- Mock wrapper tests pass.
- If Isaac Lab is available locally, the script can reset and step the environment.

---

## Task 003 — PR-2 Observation Adapter

Status: TODO

### Goal

Convert raw Isaac Lab observations / scene tensors into `ObsPacket`.

### Allowed Files

```text
go1_lewm_mpc/envs/obs_adapter.py
go1_lewm_mpc/tests/fixtures.py
go1_lewm_mpc/tests/test_obs_adapter.py
README.md
```

### Requirements

Implement:

```python
class ObsAdapter:
    def __init__(self, foot_order=("FL", "FR", "RL", "RR")):
        ...

    def from_isaac(self, raw_obs, env, env_id: int = 0) -> ObsPacket:
        ...
```

Must support fake data from `tests/fixtures.py`.

Fallback policy:

- Missing foot positions → zero array + warning.
- Missing height scan → `None`.
- Missing payload mass → `0.0`.
- Invalid shape → raise `ValueError`.

### Acceptance

```bash
python -m pytest go1_lewm_mpc/tests/test_obs_adapter.py -q
python -m pytest go1_lewm_mpc/tests -q
```

---

## Task 004 — PR-3 Dataset Collector

Status: TODO

### Goal

Collect baseline Go1 rollout data into HDF5 for future LEWM training.

### Allowed Files

```text
go1_lewm_mpc/data/__init__.py
go1_lewm_mpc/data/dataset_schema.py
go1_lewm_mpc/data/hdf5_writer.py
go1_lewm_mpc/data/replay_loader.py
go1_lewm_mpc/tests/test_dataset_schema.py
scripts/collect_dataset.py
README.md
```

### Requirements

Implement HDF5 schema:

```text
/episode_000000/
    t
    base_pos_w
    base_quat_wxyz
    base_lin_vel_w
    base_ang_vel_w
    joint_pos
    joint_vel
    foot_pos_b
    foot_pos_w
    foot_contact
    cmd_vel
    height_scan
    last_action
    payload_mass
    success
    fall
```

`collect_dataset.py` must support:

```bash
python scripts/collect_dataset.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --episodes 5 \
  --episode_len 200 \
  --out data/go1_rough_payload_v0.hdf5 \
  --headless
```

### Acceptance

- Fake data HDF5 write/read test passes.
- No HDF5 file is committed.
- Dataset schema has explicit shape checks.

---

## Task 005 — PR-4 DummyLEWM and World Model Interface

Status: TODO

### Goal

Implement a world model interface and a dummy risk model so the full closed loop can be tested before real LEWM is available.

### Allowed Files

```text
go1_lewm_mpc/world_model/__init__.py
go1_lewm_mpc/world_model/base.py
go1_lewm_mpc/world_model/dummy_lewm.py
go1_lewm_mpc/world_model/terrain_head.py
go1_lewm_mpc/world_model/state_head.py
go1_lewm_mpc/tests/test_dummy_lewm.py
README.md
```

### Requirements

Implement:

```python
class WorldModelBase(ABC):
    def encode(self, obs: ObsPacket) -> LatentPacket:
        ...

    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        ...

    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        ...
```

`DummyLEWM` risk behavior:

- Far from body → high risk.
- Too high/low z → high risk.
- Rough height scan → higher risk.
- Higher payload → more conservative risk.
- Nominal stance area → lower risk.

### Acceptance

- No torch dependency required.
- `predict_risk()` returns finite `[K]`.
- Fake-data tests pass.

---

## Task 006 — PR-5 Foothold Candidate Generator

Status: TODO

### Goal

Generate candidate footholds for the current swing leg.

### Allowed Files

```text
go1_lewm_mpc/foothold/__init__.py
go1_lewm_mpc/foothold/phase_estimator.py
go1_lewm_mpc/foothold/candidate_generator.py
go1_lewm_mpc/foothold/risk_map.py
go1_lewm_mpc/tests/test_candidate_generator.py
README.md
```

### Requirements

Implement:

```python
class PhaseEstimator:
    def update(self, obs: ObsPacket) -> int:
        ...

class FootholdCandidateGenerator:
    def generate(self, obs: ObsPacket, swing_leg_id: int) -> np.ndarray:
        ...
```

Candidate behavior:

- Generate `[K, 3]`.
- Respect max step x/y/z.
- Forward velocity shifts candidates forward.
- Side velocity shifts candidates sideways.
- Yaw command influences lateral/asymmetric shift.
- Fallback to fixed trot sequence if contact information is unavailable.

### Acceptance

- Tests cover all four legs.
- Forward command increases average candidate x.
- Side command changes average candidate y.
- Invalid swing leg raises `ValueError`.

---

## Task 007 — PR-6 OSQP Foothold Selector

Status: TODO

### Goal

Select one foothold from candidates using OSQP or a robust heuristic fallback.

### Allowed Files

```text
go1_lewm_mpc/mpc/__init__.py
go1_lewm_mpc/mpc/osqp_foothold.py
go1_lewm_mpc/mpc/cost_terms.py
go1_lewm_mpc/mpc/constraints.py
go1_lewm_mpc/foothold/selector.py
go1_lewm_mpc/tests/test_osqp_foothold.py
README.md
```

### Requirements

Implement:

```python
class OSQPFootholdSelector:
    def select(
        self,
        obs: ObsPacket,
        swing_leg_id: int,
        candidates_b: np.ndarray,
        risk: np.ndarray,
    ) -> MpcPlanPacket:
        ...
```

Fallback behavior:

- If OSQP fails, use `np.argmin(total_score)`.
- Record solver status in `plan.debug`.
- Lower confidence on fallback.

### Acceptance

- High-risk candidates are avoided.
- Equal-risk case selects near nominal point.
- OSQP solve time is recorded.
- Fallback works when OSQP is unavailable or fails.

---

## Task 008 — PR-7 Cue Injection

Status: TODO

### Goal

Convert selected foothold plan into a safe low-level command cue.

### Allowed Files

```text
go1_lewm_mpc/controllers/__init__.py
go1_lewm_mpc/controllers/cue_injection.py
go1_lewm_mpc/controllers/command_filter.py
go1_lewm_mpc/controllers/low_level_policy_wrapper.py
go1_lewm_mpc/tests/test_cue_injection.py
README.md
```

### Requirements

Implement velocity-bias cue:

```python
def foothold_to_velocity_bias(
    obs: ObsPacket,
    plan: MpcPlanPacket,
    gain_xy: float,
    gain_yaw: float,
    max_bias: np.ndarray,
) -> np.ndarray:
    ...
```

Implement:

```python
class CommandFilter:
    def update(self, cmd: np.ndarray) -> np.ndarray:
        ...

class LowLevelPolicyWrapper:
    def compute_action(self, raw_obs, cue: LowLevelCue):
        ...
```

### Acceptance

- Corrected command equals command plus clipped bias.
- No NaNs.
- Cue disabled returns baseline behavior.
- Fake policy test passes.

---

## Task 009 — PR-8 Closed-loop Runner

Status: TODO

### Goal

Run the first complete dummy closed loop.

### Allowed Files

```text
scripts/run_closed_loop.py
go1_lewm_mpc/eval/__init__.py
go1_lewm_mpc/eval/metrics.py
go1_lewm_mpc/tests/test_closed_loop_smoke.py
README.md
```

### Requirements

Support:

```bash
python scripts/run_closed_loop.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --duration_sec 10 \
  --world_model dummy \
  --use_mpc true \
  --use_cue true \
  --headless
```

Must support modes:

```text
baseline: --use_mpc false --use_cue false
no-cue:   --use_mpc true  --use_cue false
cue:      --use_mpc true  --use_cue true
```

### Acceptance

- Fake closed-loop smoke test passes.
- Logs selected foothold, risk, bias, fall/base height if available.
- NaN detection stops loop and saves debug dump.
- Isaac Lab unavailable case is clear and graceful.

---

## Task 010 — PR-9 Evaluation Scripts

Status: TODO

### Goal

Provide repeatable comparison between baseline, heuristic, dummy LEWM, and real LEWM modes.

### Allowed Files

```text
scripts/eval_closed_loop.py
go1_lewm_mpc/eval/benchmark_payload.py
go1_lewm_mpc/eval/benchmark_terrain.py
go1_lewm_mpc/eval/metrics.py
configs/eval/benchmark.yaml
go1_lewm_mpc/tests/test_metrics.py
README.md
```

### Required Metrics

```text
success_rate
fall_rate
mean_episode_length
base_height_min
body_roll_rms
body_pitch_rms
velocity_tracking_error
slip_proxy
mean_risk_selected
mean_risk_available
mpc_solve_time_mean_ms
mpc_solve_time_p95_ms
cue_norm_mean
```

### Acceptance

- Outputs:
  - `metrics.csv`
  - `summary.json`
  - `config.yaml`
- Baseline and cue results are separated.
- If a metric cannot be computed, record NaN with explanation, not fake values.

---

## Task 011 — PR-10 Real LEWM Adapter

Status: TODO

### Goal

Add a real LEWM adapter behind the existing `WorldModelBase` interface.

### Allowed Files

```text
go1_lewm_mpc/world_model/lewm_adapter.py
scripts/train_lewm.py
configs/lewm/train_lewm.yaml
go1_lewm_mpc/tests/test_lewm_adapter_mock.py
README.md
```

### Requirements

Implement:

```python
class LEWMAdapter(WorldModelBase):
    def __init__(self, checkpoint_path: str, cfg: dict, device: str = "cuda"):
        ...

    def encode(self, obs: ObsPacket) -> LatentPacket:
        ...

    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        ...

    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        ...
```

### Acceptance

- Can replace `DummyLEWM` without modifying closed-loop runner.
- Missing checkpoint gives clear error.
- CPU fallback works with warning.
- Risk output shape is `[K]` and finite.

---

## Task 012 — PR-11 Ablation Modes

Status: TODO

### Goal

Add clean ablation modes.

### Allowed Files

```text
scripts/eval_closed_loop.py
go1_lewm_mpc/eval/metrics.py
configs/eval/benchmark.yaml
README.md
```

### Required Modes

```text
baseline
heuristic_only
dummy_lewm_risk
lewm_risk
lewm_risk_no_payload
lewm_risk_no_height
```

### Acceptance

- Each mode is selectable.
- Unimplemented mode raises `NotImplementedError`.
- `ablation_summary.csv` is generated.

# Go1 LEWM MPC

For Codex: read `AGENTS.md` before making changes. This is an existing repository, not a greenfield project.

Phase 1 builds a mock-testable scaffold for a Unitree Go1 foothold-cue system in Isaac Lab. The intended architecture keeps the official Go1 rough-terrain locomotion policy as the low-level controller, while a future LEWM layer predicts local terrain risk and reduced-order state features for foothold selection.

```text
Isaac Lab Go1 rough env
    -> ObsAdapter
    -> LEWM / DummyLEWM
    -> terrain risk and short-horizon prediction
    -> heuristic candidate generation
    -> OSQP or heuristic foothold selection
    -> velocity bias / observation cue
    -> official Go1 locomotion policy
```

The current baseline wrapper intentionally keeps Isaac Lab imports lazy. Importing the Python package and running mock tests does not start Omniverse or require Isaac Lab.

## Baseline Go1 Runner

Show CLI options:

```bash
python scripts/run_baseline.py --help
```

Run the official Go1 rough velocity task when Isaac Lab is available:

```bash
python scripts/run_baseline.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --headless \
  --duration_sec 5
```

If plain Python cannot launch Isaac Lab in your setup, use the Isaac Lab launcher:

```bash
./isaaclab.sh -p scripts/run_baseline.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --headless \
  --duration_sec 5
```

Task 002 does not add ObsAdapter, dataset collection, LEWM, MPC, cue injection, or a closed-loop runner.

## Observation Adapter

The adapter converts explicit raw observation fields or Isaac-like scene tensors into `ObsPacket`:

```python
from go1_lewm_mpc.envs.obs_adapter import ObsAdapter

packet = ObsAdapter().from_isaac(raw_obs, env, env_id=0)
```

Missing foot positions use zero arrays with warnings, missing height scans become `None`, and missing payload mass defaults to `0.0`. Invalid core shapes raise `ValueError`.

## Dataset Collection

Task 004 adds a HDF5 schema for baseline rollouts. Each episode is stored as `/episode_000000/...` with time-series robot state, command, terrain scan, payload mass, and scalar `success` / `fall` flags.

```bash
python scripts/collect_dataset.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --episodes 5 \
  --episode_len 200 \
  --out data/go1_rough_payload_v0.hdf5 \
  --headless
```

The collector uses the baseline wrapper and `ObsAdapter`; it does not train LEWM or run MPC.

## Dummy World Model

Task 005 adds `WorldModelBase` and a deterministic `DummyLEWM` implementation:

```python
from go1_lewm_mpc.world_model import DummyLEWM

latent = DummyLEWM().encode(obs)
risk = DummyLEWM().predict_risk(obs, query_points_b)
pred_state = DummyLEWM().predict_state(obs, horizon=10, dt=0.02)
```

`DummyLEWM` is rule based and has no torch dependency. It is only a smoke-testable stand-in for real LEWM.

## LEWM Adapter

Task 011 adds `LEWMAdapter`, a torch-backed adapter with the same `WorldModelBase` methods as `DummyLEWM`:

```python
from go1_lewm_mpc.world_model.lewm_adapter import LEWMAdapter

model = LEWMAdapter(
    checkpoint_path="checkpoints/lewm_adapter_v0.ckpt",
    cfg={"latent_dim": 16},
    device="cuda",
)
risk = model.predict_risk(obs, query_points_b)
pred_state = model.predict_state(obs, horizon=10, dt=0.02)
```

The adapter raises a clear `FileNotFoundError` when the checkpoint path is missing. If `device="cuda"` is requested but CUDA is unavailable, it warns and falls back to CPU. The first checkpoint contract is intentionally small: metadata-only mock checkpoints, linear heads, and risk MLP state dicts are supported so the closed-loop code can swap in a learned model without changing the `WorldModelBase` call sites.

The training entrypoint is present for config validation:

```bash
python scripts/train_lewm.py --config configs/lewm/train_lewm.yaml --dry_run
```

Full LEWM training is deferred; running without `--dry_run` raises `NotImplementedError` instead of silently producing a placeholder checkpoint.

## Foothold Candidates

Task 006 adds a contact/fallback phase estimator and deterministic candidate generator:

```python
from go1_lewm_mpc.foothold import FootholdCandidateGenerator, PhaseEstimator

swing_leg_id = PhaseEstimator().update(obs)
candidates_b = FootholdCandidateGenerator().generate(obs, swing_leg_id)
```

Candidates are body-frame `[K, 3]` points clipped to a per-leg reachable box. Velocity commands shift the candidate center; no foothold selection or MPC is performed in this task.

## Foothold Selection

Task 007 adds `OSQPFootholdSelector`, which solves a small OSQP reference problem and projects the result back to the nearest candidate. If OSQP is unavailable or fails, it falls back to the lowest total heuristic score and records the reason in `plan.debug`.

```python
from go1_lewm_mpc.mpc import OSQPFootholdSelector

plan = OSQPFootholdSelector().select(obs, swing_leg_id, candidates_b, risk)
```

The selector returns an `MpcPlanPacket` with a selected foothold and zero velocity bias. Converting the plan into a low-level command cue is left to the next task.

## Cue Injection

Task 008 converts a selected foothold into a clipped velocity-bias cue:

```python
from go1_lewm_mpc.controllers import make_low_level_cue, LowLevelPolicyWrapper

cue = make_low_level_cue(obs, plan)
action = LowLevelPolicyWrapper(policy, use_cue=True).compute_action(raw_obs, cue)
```

The first version only supports soft command correction. It does not change the low-level policy network or add a closed-loop runner.

## Closed-Loop Smoke Runner

Task 009 wires the mock-testable pipeline together:

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

Supported modes are baseline (`--use_mpc false --use_cue false`), no-cue (`--use_mpc true --use_cue false`), and cue (`--use_mpc true --use_cue true`). The runner logs selected footholds, risk, velocity bias, fall state, and base height. It stops on NaNs and writes a debug dump.

## Evaluation

Task 010 adds repeatable episode-level evaluation outputs:

```bash
python scripts/eval_closed_loop.py \
  --config configs/eval/benchmark.yaml \
  --out_dir runs/example_scheme2_eval
```

For mock-only validation without Isaac Lab:

```bash
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04
```

Each run writes `config.yaml`, `metrics.csv`, `summary.json`, and `ablation_summary.csv`. PR-11 declares the ablation vocabulary used by the project:

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

The default benchmark config runs only modes that the current closed-loop runner can express without pretending to use unavailable LeWM paths: `baseline` and `dummy_risk`. Modes that still need runner support or a real checkpoint raise `NotImplementedError` instead of silently falling back to baseline. Metrics that cannot be computed, such as the current slip proxy without foot velocity, are recorded as NaN with explanations instead of fabricated values.

## Quick Check

```bash
python -m pytest go1_lewm_mpc/tests -q
python -c "from go1_lewm_mpc.common.types import ObsPacket"
```

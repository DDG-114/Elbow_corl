# Elbow_corl — Codex Agentic Coding Checklist

> 适用仓库：`https://github.com/DDG-114/Elbow_corl`  
> 目标：把当前 Go1 + Isaac Lab + MPC foothold-cue 原型，重构为更贴近 `lucas-maes/le-wm` 原始设计的世界模型项目。  
> 核心方向：**LeWM 是 latent dynamics world model，不是直接的 foothold risk predictor。当前的 risk / terrain / state head 应降级为 auxiliary probe head。**

---

## 0. 给 Codex 的总原则

Codex 必须先读：

```text
AGENTS.md
TASKS.md
CODEX_TASK_QUEUE.md
README.md
```

如果根目录没有 `AGENTS.md` / `CODEX_TASK_QUEUE.md` / `CODEX_PROMPTS.md`，先执行 PR-00。

### 0.1 不允许做的事

```text
- 不要从零重写整个仓库。
- 不要删除已经通过测试的 Phase 1 scaffold。
- 不要让 LEWM 直接输出 12D joint action。
- 不要把 foothold risk head 当作核心 world model。
- 不要把 upstream le-wm checkpoint 假装成已经兼容 Go1。
- 不要把测试 fixture 作为生产 runtime dependency。
- 不要在没有 Isaac Lab 的环境中强制跑仿真。
- 不要静默 fallback 到 baseline 伪造实验结果。
```

### 0.2 必须遵守的 LeWM 语义

上游 `lucas-maes/le-wm` 的核心语义是：

```text
observation frame o_t
    ↓
encoder
    ↓
latent z_t
    ↓ + action a_t
predictor
    ↓
predicted next latent z_{t+1}
```

因此本仓库中的世界模型应调整为：

```text
Go1 local observation frame / heightmap / visual-like terrain patch
    ↓
LeWM encoder
    ↓
latent z_t
    ↓ + high-level action / MidAction / command sequence
LeWM predictor rollout
    ↓
future latent sequence
    ↓
planning score / auxiliary probes
    ↓
MPC / cue generation
```

也就是说：

```text
核心世界模型职责：encode + latent rollout + latent planning score
辅助职责：risk probe / state probe / terrain probe
控制职责：仍由 MPC / cue injection / low-level policy 完成
```

---

## 1. 当前仓库判断

当前仓库已经具备 Phase 1 大部分脚手架：

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

所以 Codex 后续任务是 **增量重构**，不是 greenfield implementation。

当前主要 mismatch：

| 当前仓库 | LeWM 原始设计 | 需要调整 |
|---|---|---|
| `LEWMAdapter.predict_risk(obs, candidates)` 是核心接口 | LeWM 核心是 encoder + predictor latent dynamics | risk 改为 auxiliary probe |
| 输入是 `ObsPacket` 里的 proprio + height_scan | LeWM 原始输入是 frame / pixels | 构造 `WorldModelInputFrame`，先用 2D heightmap 作为 visual-like frame |
| `predict_state()` 是 constant velocity fallback | LeWM predictor 预测 next latent | 新增 `rollout_latent()` |
| MPC 只用 one-step candidate risk | LeWM planning 用 latent rollout + CEM | 新增 latent planner / action sequence scorer |
| `run_closed_loop.py` 只允许 dummy world model | 需要可切换 dummy / local_lewm / upstream_lewm | 新增 world model factory |
| `eval_closed_loop.py` import tests fixture | runtime 不能依赖 tests | fake env 移到 runtime mock 模块 |

---

## 2. 推荐 PR 队列

> 每次只让 Codex 做一个 PR。  
> 每个 PR 完成后必须跑 focused test + full lightweight test。  
> Isaac Lab 集成只作为本地附加验证，不作为 cloud Codex 的硬门槛。

---

## PR-00 — Agentic Docs Normalization

### 目标

把 agentic coding 规则提升到仓库根目录，并更新任务状态，防止 Codex 重复实现已经存在的模块。

### 允许修改

```text
AGENTS.md
CODEX_TASK_QUEUE.md
CODEX_PROMPTS.md
README.md
.gitignore
```

### 具体任务

1. 如果根目录不存在 `AGENTS.md`，从 `go1_codex_agentic_pack/AGENTS.md` 复制并更新。
2. 如果根目录不存在 `CODEX_TASK_QUEUE.md`，新建一份 **当前仓库状态版**。
3. 将任务状态设为：
   ```text
   Task 001-008: DONE
   Task 009-011: PARTIAL
   Task 012: TODO
   New LeWM-alignment PRs: TODO
   ```
4. README 增加：
   ```text
   For Codex: read AGENTS.md before making changes.
   This is not a greenfield repository.
   ```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_types.py -q
python -c "from go1_lewm_mpc.common.types import ObsPacket; print('ok')"
```

### Codex Prompt

```text
Read README.md, TASKS.md, and go1_codex_agentic_pack/AGENTS.md.

Implement PR-00 only.

This repository already contains most Phase 1 scaffold. Do not rewrite existing modules.
Create root-level AGENTS.md, CODEX_TASK_QUEUE.md, and CODEX_PROMPTS.md if missing.
Update task statuses to reflect actual implementation state:
- Task 001-008 DONE
- Task 009-011 PARTIAL
- Task 012 TODO
- New LeWM-alignment PRs TODO

Allowed files:
AGENTS.md
CODEX_TASK_QUEUE.md
CODEX_PROMPTS.md
README.md
.gitignore

Do not modify Python runtime code.

Run:
python -m pytest go1_lewm_mpc/tests/test_types.py -q
python -c "from go1_lewm_mpc.common.types import ObsPacket; print('ok')"

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-01 — Runtime Mock Cleanup

### 目标

去掉 runtime scripts 对 `go1_lewm_mpc.tests.fixtures` 的依赖。

### 允许修改

```text
go1_lewm_mpc/mock/__init__.py
go1_lewm_mpc/mock/fake_isaac_env.py
go1_lewm_mpc/tests/fixtures.py
scripts/eval_closed_loop.py
go1_lewm_mpc/tests/test_closed_loop_smoke.py
go1_lewm_mpc/tests/test_metrics.py
```

### 具体任务

1. 新增 runtime-safe mock 模块：
   ```text
   go1_lewm_mpc/mock/fake_isaac_env.py
   ```
2. 把 `FakeIsaacEnv` 从 `tests/fixtures.py` 移到 runtime mock。
3. tests 可以 import runtime mock，但 runtime code 不允许 import tests。
4. 修改 `scripts/eval_closed_loop.py`：
   ```python
   from go1_lewm_mpc.mock.fake_isaac_env import FakeIsaacEnv
   ```
5. 添加测试，确保：
   ```text
   scripts/eval_closed_loop.py 中不包含 "go1_lewm_mpc.tests"
   ```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q
python -m pytest go1_lewm_mpc/tests/test_metrics.py -q
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-01 only: remove runtime dependency on tests fixtures.

Goal:
Move FakeIsaacEnv into go1_lewm_mpc/mock/fake_isaac_env.py and update scripts/eval_closed_loop.py to import from the runtime-safe mock module.

Do not modify:
- world_model
- mpc
- controllers
- ObsPacket schema

Acceptance:
1. No runtime file imports go1_lewm_mpc.tests.*
2. pytest closed-loop smoke passes.
3. eval fake mode runs.

Run:
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-02 — World Model Input Frame Contract

### 目标

让 Go1 observation 可以转换成 LeWM 风格的 `observation frame`，而不是只作为 proprio dict。

### 允许修改

```text
go1_lewm_mpc/common/types.py
go1_lewm_mpc/world_model/input_frame.py
go1_lewm_mpc/envs/obs_adapter.py
go1_lewm_mpc/tests/fixtures.py
go1_lewm_mpc/tests/test_obs_adapter.py
go1_lewm_mpc/tests/test_world_model_input_frame.py
configs/env/go1_lewm_rough.yaml
```

### 新增数据结构

```python
@dataclass
class WorldModelInputFrame:
    """
    Visual-like frame input for LeWM-style encoder.

    This is not raw RGB in Phase 1.
    It can be a local heightmap frame with shape [C, H, W].
    """
    t: float
    frame: np.ndarray          # shape [C, H, W], float32
    frame_type: str            # "heightmap", "depth", "rgb"
    action_context: np.ndarray # optional high-level command / previous action
    metadata: dict
```

### 新增函数

```python
def obs_to_heightmap_frame(
    obs: ObsPacket,
    size: tuple[int, int] = (64, 64),
    normalize: bool = True,
) -> WorldModelInputFrame:
    ...
```

### 约定

```text
height_scan shape [Nh]  -> reshape/interpolate to [1, H, W]
height_scan shape [H,W] -> convert to [1, H, W]
height_scan None        -> zero frame + metadata["missing_height_scan"] = True
```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_world_model_input_frame.py -q
python -m pytest go1_lewm_mpc/tests/test_obs_adapter.py -q
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-02 only: add a LeWM-style WorldModelInputFrame contract.

Goal:
Do not redesign ObsPacket. Add a conversion layer that turns ObsPacket.height_scan into a visual-like [C,H,W] frame usable by a LeWM encoder.

Allowed files:
go1_lewm_mpc/common/types.py
go1_lewm_mpc/world_model/input_frame.py
go1_lewm_mpc/tests/fixtures.py
go1_lewm_mpc/tests/test_world_model_input_frame.py
go1_lewm_mpc/tests/test_obs_adapter.py
configs/env/go1_lewm_rough.yaml

Requirements:
- Support height_scan None, [Nh], and [H,W].
- Output frame shape must be [1,64,64] by default.
- No torch dependency.
- No Isaac Lab dependency.

Run:
python -m pytest go1_lewm_mpc/tests/test_world_model_input_frame.py -q
python -m pytest go1_lewm_mpc/tests/test_obs_adapter.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-03 — LeWM Semantic Interface Refactor

### 目标

把 `WorldModelBase` 从 task-specific risk predictor 调整为 LeWM-style latent dynamics interface，同时保留兼容层。

### 允许修改

```text
go1_lewm_mpc/world_model/base.py
go1_lewm_mpc/world_model/dummy_lewm.py
go1_lewm_mpc/world_model/lewm_adapter.py
go1_lewm_mpc/world_model/input_frame.py
go1_lewm_mpc/tests/test_dummy_lewm.py
go1_lewm_mpc/tests/test_lewm_semantic_interface.py
```

### 新接口

`WorldModelBase` 应包含：

```python
class WorldModelBase(ABC):
    def encode(self, obs: ObsPacket) -> LatentPacket:
        ...

    def encode_frame(self, frame: WorldModelInputFrame) -> LatentPacket:
        ...

    def predict_next_latent(
        self,
        latent: LatentPacket,
        action: np.ndarray,
    ) -> LatentPacket:
        ...

    def rollout_latent(
        self,
        obs: ObsPacket,
        action_sequence: np.ndarray,
        dt: float,
    ) -> list[LatentPacket]:
        ...

    # Auxiliary probe heads, not core world model semantics.
    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        ...

    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        ...
```

### 重要语义

```text
encode / predict_next_latent / rollout_latent 是核心世界模型功能。
predict_risk / predict_state 是 auxiliary probe。
```

必须在 docstring 里写清楚。

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_dummy_lewm.py -q
python -m pytest go1_lewm_mpc/tests/test_lewm_semantic_interface.py -q
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-03 only: refactor WorldModelBase semantics toward LeWM-style latent dynamics.

Do not delete predict_risk or predict_state, but mark them as auxiliary probe methods.
Add encode_frame(), predict_next_latent(), and rollout_latent().

DummyLEWM must implement all methods without torch.
LEWMAdapter may implement simple placeholder latent transition if needed, but must not pretend to be a true upstream le-wm bridge yet.

Run:
python -m pytest go1_lewm_mpc/tests/test_dummy_lewm.py -q
python -m pytest go1_lewm_mpc/tests/test_lewm_semantic_interface.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-04 — High-Level Action / MidAction Contract

### 目标

为 LeWM predictor 定义 action conditioning，不再把 action 含糊地等同于 low-level 12D joint action。

### 允许修改

```text
go1_lewm_mpc/common/types.py
go1_lewm_mpc/world_model/action_adapter.py
go1_lewm_mpc/tests/test_action_adapter.py
```

### 新增数据结构

```python
@dataclass
class MidAction:
    """
    High-level action used by LeWM predictor.

    This is NOT 12D joint action.
    It represents command-level or foothold-cue-level action.
    """
    t: float
    cmd_vel: np.ndarray          # [3]
    velocity_bias: np.ndarray    # [3]
    selected_leg_id: int | None
    foothold_delta_b: np.ndarray | None  # [3]
```

### 新增函数

```python
def plan_to_mid_action(obs: ObsPacket, plan: MpcPlanPacket | None) -> MidAction:
    ...

def mid_action_to_vector(action: MidAction) -> np.ndarray:
    ...
```

### 建议 action vector

```text
[vx, vy, yaw_rate, dvx, dvy, dyaw, selected_leg_onehot(4), foothold_delta_xyz(3)]
总维度：13
```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_action_adapter.py -q
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-04 only: add MidAction and action adapter for LeWM action conditioning.

Rules:
- MidAction is high-level.
- Do not use 12D joint action as LeWM action.
- Do not modify low-level policy.

Run:
python -m pytest go1_lewm_mpc/tests/test_action_adapter.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-05 — World Model Factory

### 目标

让 `run_closed_loop.py` 可以通过统一 factory 切换：

```text
dummy
local_lewm
upstream_lewm_mock
```

### 允许修改

```text
go1_lewm_mpc/world_model/factory.py
go1_lewm_mpc/world_model/lewm_adapter.py
scripts/run_closed_loop.py
scripts/eval_closed_loop.py
configs/lewm/train_lewm.yaml
go1_lewm_mpc/tests/test_world_model_factory.py
go1_lewm_mpc/tests/test_closed_loop_smoke.py
```

### 新增接口

```python
def build_world_model(
    backend: str,
    cfg: dict,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> WorldModelBase:
    ...
```

### backend 行为

```text
dummy:
  returns DummyLEWM

local_lewm:
  returns current LEWMAdapter using local checkpoint contract

upstream_lewm_mock:
  returns a mock upstream bridge that supports encode_frame and rollout_latent
  but raises NotImplementedError for real upstream checkpoint loading
```

### 修改 CLI

`scripts/run_closed_loop.py` 支持：

```bash
--world_model dummy
--world_model local_lewm
--world_model upstream_lewm_mock
--world_model_ckpt path/to.ckpt
--world_model_cfg configs/lewm/train_lewm.yaml
```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_world_model_factory.py -q
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-05 only: add world_model/factory.py and remove hard-coded dummy-only limitation from run_closed_loop.py.

Do not implement real upstream le-wm loading yet.
Add backend choices:
dummy
local_lewm
upstream_lewm_mock

Unsupported real upstream loading should raise NotImplementedError clearly.

Run:
python -m pytest go1_lewm_mpc/tests/test_world_model_factory.py -q
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-06 — Upstream LeWM Bridge Skeleton

### 目标

建立真正对接 `lucas-maes/le-wm` 的边界层，但不要求立即训练或跑通真实 checkpoint。

### 允许修改

```text
go1_lewm_mpc/world_model/upstream_lewm_bridge.py
go1_lewm_mpc/world_model/lewm_adapter.py
go1_lewm_mpc/world_model/factory.py
go1_lewm_mpc/tests/test_upstream_lewm_bridge.py
configs/lewm/upstream_lewm.yaml
README.md
```

### 新增类

```python
class UpstreamLeWMBridge(WorldModelBase):
    """
    Bridge for lucas-maes/le-wm style encoder-predictor.

    This class adapts local Go1 WorldModelInputFrame and MidAction
    into upstream LeWM-style frame/action inputs.
    """

    def __init__(
        self,
        upstream_repo: str | None,
        checkpoint_path: str | None,
        cfg: dict,
        device: str = "cuda",
        allow_mock: bool = False,
    ):
        ...

    def encode_frame(self, frame: WorldModelInputFrame) -> LatentPacket:
        ...

    def predict_next_latent(self, latent: LatentPacket, action: np.ndarray) -> LatentPacket:
        ...

    def rollout_latent(self, obs: ObsPacket, action_sequence: np.ndarray, dt: float) -> list[LatentPacket]:
        ...
```

### 行为要求

```text
allow_mock=True:
  使用 deterministic mock encoder/predictor，测试可跑。

allow_mock=False 且 upstream repo / checkpoint 缺失:
  抛 NotImplementedError 或 FileNotFoundError，不静默 fallback。

真实 upstream import:
  lazy import，不允许在 module import 时失败。
```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_upstream_lewm_bridge.py -q
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-06 only: add UpstreamLeWMBridge skeleton.

Important:
This PR does NOT need to run real lucas-maes/le-wm.
It must define a clean boundary and mock mode.
No import of upstream le-wm at module import time.
Real mode may raise clear NotImplementedError.

Run:
python -m pytest go1_lewm_mpc/tests/test_upstream_lewm_bridge.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-07 — LeWM Dataset Sequence Schema

### 目标

当前 HDF5 rollout schema 是 robot-state oriented。需要补充 LeWM training sequence schema。

### 允许修改

```text
go1_lewm_mpc/data/dataset_schema.py
go1_lewm_mpc/data/hdf5_writer.py
go1_lewm_mpc/data/replay_loader.py
go1_lewm_mpc/data/lewm_sequence_dataset.py
scripts/collect_dataset.py
go1_lewm_mpc/tests/test_lewm_sequence_dataset.py
```

### 新增 LeWM sequence schema

每个 episode 增加或派生：

```text
/world_model/
    frame                 [T, C, H, W]
    action                [T, A]
    next_frame            [T, C, H, W]
    done                  [T]
    probe/base_state      [T, D_state]
    probe/foothold_risk   [T, 4] or optional
    probe/payload_mass    [T, 1]
```

### 新增 loader

```python
class LeWMSequenceDataset:
    def __init__(self, hdf5_path: str, seq_len: int, frame_key: str = "world_model/frame"):
        ...

    def __getitem__(self, index: int) -> dict:
        return {
            "frame": ...,        # [L, C, H, W]
            "action": ...,       # [L, A]
            "next_frame": ...,   # [L, C, H, W]
            "probe": {...},
        }
```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_lewm_sequence_dataset.py -q
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-07 only: add LeWM sequence dataset schema.

Goal:
Create a dataset view suitable for LeWM-style training:
frame o_t, action a_t, next_frame o_{t+1}, optional probe labels.

Do not implement neural training yet.
Do not depend on torch.
Fake HDF5 write/read test must pass.

Run:
python -m pytest go1_lewm_mpc/tests/test_lewm_sequence_dataset.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-08 — LeWM Loss and Training Dry Run

### 目标

加入 LeWM 风格训练结构：prediction loss + SIGReg regularizer；先实现 dry-run 和 mock training step。

### 允许修改

```text
go1_lewm_mpc/world_model/lewm_loss.py
go1_lewm_mpc/world_model/simple_lewm_backbone.py
scripts/train_lewm.py
configs/lewm/train_lewm.yaml
go1_lewm_mpc/tests/test_lewm_loss.py
go1_lewm_mpc/tests/test_train_lewm_dry_run.py
```

### 新增函数

```python
def latent_prediction_loss(pred_z: torch.Tensor, target_z: torch.Tensor) -> torch.Tensor:
    ...

def sigreg_loss(z: torch.Tensor, eps: float = 1e-4) -> torch.Tensor:
    ...

def lewm_total_loss(pred_z, target_z, batch_z, lambda_sigreg: float) -> dict:
    ...
```

### 训练脚本要求

```bash
python scripts/train_lewm.py --config configs/lewm/train_lewm.yaml --dry_run
```

必须输出：

```text
dataset path
frame shape
action dim
latent dim
lambda_sigreg
loss keys
```

非 dry-run 可以先继续 `NotImplementedError`，但必须清楚说明下一步。

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_lewm_loss.py -q
python scripts/train_lewm.py --config configs/lewm/train_lewm.yaml --dry_run
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-08 only: add LeWM-style loss functions and training dry-run.

Important:
This PR should align with LeWM semantics:
loss = latent prediction loss + lambda * SIGReg.
Do not claim full upstream le-wm training is complete.

Run:
python -m pytest go1_lewm_mpc/tests/test_lewm_loss.py -q
python scripts/train_lewm.py --config configs/lewm/train_lewm.yaml --dry_run

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-09 — Latent CEM Planner Prototype

### 目标

增加 LeWM 原始设计中的 planning 方式：latent rollout + CEM action sequence search。它不直接控制关节，只输出 high-level action sequence / suggested MidAction。

### 允许修改

```text
go1_lewm_mpc/world_model/latent_planner.py
go1_lewm_mpc/world_model/action_adapter.py
go1_lewm_mpc/tests/test_latent_planner.py
```

### 新增类

```python
class LatentCEMPlanner:
    def __init__(
        self,
        world_model: WorldModelBase,
        action_dim: int,
        horizon: int,
        population: int,
        elite_frac: float,
        iterations: int,
        action_bounds: tuple[np.ndarray, np.ndarray],
    ):
        ...

    def plan(
        self,
        obs: ObsPacket,
        goal_latent: LatentPacket | None = None,
        scoring_fn: Callable | None = None,
    ) -> np.ndarray:
        """
        Return best high-level action sequence with shape [H, A].
        """
```

### 默认 scoring

如果没有 goal image/latent，使用 surrogate scoring：

```text
low predicted uncertainty
low auxiliary risk
stable reduced-order state
small action magnitude
```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_latent_planner.py -q
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-09 only: add LatentCEMPlanner.

This planner must use WorldModelBase.rollout_latent().
It must output high-level action sequences, not 12D joint actions.
Use deterministic random seed in tests.

Run:
python -m pytest go1_lewm_mpc/tests/test_latent_planner.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-10 — MPC Cost Uses Latent Rollout

### 目标

让 OSQP foothold selector 不再只看 `risk`，而是可以接收 LeWM latent rollout score。

### 允许修改

```text
go1_lewm_mpc/mpc/osqp_foothold.py
go1_lewm_mpc/mpc/cost_terms.py
go1_lewm_mpc/world_model/latent_planner.py
scripts/run_closed_loop.py
go1_lewm_mpc/tests/test_osqp_foothold.py
go1_lewm_mpc/tests/test_closed_loop_smoke.py
```

### 新增 cost terms

```python
def latent_rollout_cost(
    latent_sequence: list[LatentPacket],
    uncertainty_weight: float,
    smoothness_weight: float,
) -> float:
    ...
```

### Selector 接口扩展

```python
def select(
    self,
    obs: ObsPacket,
    swing_leg_id: int,
    candidates_b: np.ndarray,
    risk: np.ndarray | None = None,
    latent_cost: np.ndarray | None = None,
) -> MpcPlanPacket:
    ...
```

### 语义

```text
risk 是 auxiliary probe cost。
latent_cost 是 LeWM core rollout cost。
两者可以同时存在，但 latent_cost 应该是更接近 LeWM 原始设计的主信号。
```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_osqp_foothold.py -q
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-10 only: allow MPC/selector to consume latent rollout cost.

Do not remove existing risk fallback.
Do not implement real robot hardware.
Do not modify low-level policy.

Run:
python -m pytest go1_lewm_mpc/tests/test_osqp_foothold.py -q
python -m pytest go1_lewm_mpc/tests/test_closed_loop_smoke.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-11 — Evaluation and Ablation Alignment

### 目标

建立能回答研究问题的 ablation，而不是只比较 baseline/cue。

### 允许修改

```text
scripts/eval_closed_loop.py
configs/eval/benchmark.yaml
go1_lewm_mpc/eval/metrics.py
go1_lewm_mpc/tests/test_metrics.py
README.md
```

### 必须支持 mode

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

### 输出

```text
metrics.csv
summary.json
ablation_summary.csv
config.yaml
```

### 未实现模式

必须：

```python
raise NotImplementedError("mode X is declared but not implemented")
```

不能静默 fallback 到 baseline。

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_metrics.py -q
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-11 only: evaluation and ablation alignment.

Add modes:
baseline
heuristic_only
dummy_risk
local_lewm_aux_risk
local_lewm_latent_cost
upstream_lewm_mock_latent_cost
lewm_no_payload
lewm_no_heightmap

Unimplemented declared modes must raise NotImplementedError.

Run:
python -m pytest go1_lewm_mpc/tests/test_metrics.py -q
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## PR-12 — Payload Randomization Bridge

### 目标

让 payload robustness 不只是 config 名字，而是真正进入仿真、ObsPacket、dataset 和 evaluation。

### 允许修改

```text
go1_lewm_mpc/envs/payload_randomization.py
go1_lewm_mpc/envs/go1_env_wrapper.py
go1_lewm_mpc/envs/obs_adapter.py
scripts/collect_dataset.py
scripts/run_closed_loop.py
scripts/eval_closed_loop.py
configs/env/go1_lewm_rough.yaml
configs/eval/benchmark.yaml
go1_lewm_mpc/tests/test_payload_randomization.py
```

### 新增类

```python
@dataclass
class PayloadSpec:
    mass_kg: float
    com_b: np.ndarray  # [3]

class PayloadRandomizer:
    def sample(self, rng: np.random.Generator) -> PayloadSpec:
        ...

    def apply(self, env, spec: PayloadSpec, env_ids=None) -> None:
        """
        Isaac Lab implementation may be best-effort.
        If unavailable, raise NotImplementedError in real mode.
        Fake mode should record payload spec in env metadata.
        """
```

### 验收

```bash
python -m pytest go1_lewm_mpc/tests/test_payload_randomization.py -q
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04
```

### Codex Prompt

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-12 only: payload randomization bridge.

Goal:
Payload should propagate through fake env, ObsPacket, dataset/eval metadata.
Real Isaac Lab apply() may be best-effort but must not silently pretend success.

Run:
python -m pytest go1_lewm_mpc/tests/test_payload_randomization.py -q
python scripts/eval_closed_loop.py --fake --episodes 1 --duration_sec 0.04

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## 3. 推荐执行顺序

```text
PR-00  Agentic docs normalization
PR-01  Runtime mock cleanup
PR-02  WorldModelInputFrame contract
PR-03  LeWM semantic interface refactor
PR-04  MidAction action conditioning
PR-05  World model factory
PR-06  Upstream LeWM bridge skeleton
PR-07  LeWM sequence dataset schema
PR-08  LeWM loss + training dry run
PR-09  Latent CEM planner
PR-10  MPC cost uses latent rollout
PR-11  Evaluation and ablation alignment
PR-12  Payload randomization bridge
```

优先级建议：

```text
必须先做：PR-00 ~ PR-06
然后做：PR-07 ~ PR-10
最后做：PR-11 ~ PR-12
```

---

## 4. 每次给 Codex 的固定尾部要求

每个 prompt 最后都加：

```text
Do not implement future PRs.
Do not rewrite working modules unless necessary.
Keep tests Isaac-Lab-independent unless explicitly asked.
If a requested real simulator feature cannot be implemented in this environment, add a clear NotImplementedError and a mock-mode test.

Final response format:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## 5. 人工 review checklist

每个 PR 合并前检查：

```text
[ ] 是否只完成了当前 PR？
[ ] 是否没有把 LEWM 写成直接 risk predictor？
[ ] 是否保留了 encode / latent rollout 的核心世界模型语义？
[ ] risk / terrain / state 是否被标为 auxiliary probe？
[ ] 是否没有让 LEWM 输出 12D joint actions？
[ ] 是否没有修改 low-level policy 网络？
[ ] 是否没有 runtime import tests？
[ ] 是否 mock tests 不需要 Isaac Lab？
[ ] 是否 unsupported real mode 明确 NotImplementedError？
[ ] 是否没有提交 data / runs / checkpoint / HDF5 / video？
```

---

## 6. 最推荐的第一条 Codex 指令

```text
Read README.md, TASKS.md, and go1_codex_agentic_pack/AGENTS.md.

Implement PR-00 from Elbow_corl_Codex_Agentic_Checklist.md only.

This is an existing repository, not a greenfield project.
Do not rewrite working Python modules.
Only normalize root-level agentic documentation and task status.

Run:
python -m pytest go1_lewm_mpc/tests/test_types.py -q
python -c "from go1_lewm_mpc.common.types import ObsPacket; print('ok')"

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

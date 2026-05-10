# TASKS.md — Go1 + LEWM + MPC Foothold Cue System（方案二）

> 面向 Codex 的执行任务文件  
> 项目目标：在 Isaac Sim / Isaac Lab 中，以 Unitree Go1 为平台，复用官方 Go1 rough terrain locomotion policy 作为低层控制器，在其上构建 **LEWM terrain/traversability prediction + heuristic foothold candidate generation + OSQP one-step foothold selection + low-level policy cue injection** 的快速闭环系统。  
> 核心原则：**先跑通可演示闭环，再接入真实 LEWM；先做软落脚点提示，不做硬约束 WBC。**

---

## 0. 项目边界

### 0.1 本阶段要做什么

本阶段采用“方案二”：

```text
Isaac Lab Go1 rough env
    ↓
ObsAdapter
    ↓
LEWM / DummyLEWM
    ↓
Terrain risk / traversability / short-horizon state prediction
    ↓
Heuristic foothold candidate generator
    ↓
OSQP / heuristic selector
    ↓
Velocity bias 或 observation cue
    ↓
Official Isaac Lab Go1 low-level RL policy
    ↓
Go1 motion in Isaac Sim / Isaac Lab
```

### 0.2 本阶段不做什么

Codex 不要实现以下内容：

- 不要直接让 LEWM 输出 12D joint actions。
- 不要在第一阶段实现完整 WBC / torque-level NMPC。
- 不要强行改写 Go1 低层 locomotion policy 的核心网络结构。
- 不要先接 OCS2 / Crocoddyl / full-body dynamics。
- 不要把 foothold 当作硬约束强行塞入官方 low-level policy。
- 不要依赖真实 Go1 硬件；本阶段只做 Isaac Lab 仿真闭环。
- 不要假设 LEWM 一开始已经训练好；必须提供 `DummyLEWM` / `RiskMapStub` 以支持 smoke test。

### 0.3 关键技术决策

- 低层控制器：优先复用 Isaac Lab 官方 `Isaac-Velocity-Rough-Unitree-Go1-v0` 或其导出的 RSL-RL policy。
- 世界模型：先提供接口和 dummy 实现，再接入真实 LeWM。
- MPC：先使用 OSQP 做 one-step / short-horizon foothold selection。
- 落脚点控制方式：第一阶段只作为 **soft cue**，通过 velocity correction 或 observation augmentation 影响低层。
- 目标指标：先看是否比 baseline 在 payload + rough terrain 下更稳定，而不是追求完美 footstep tracking。

---

## 1. 建议仓库结构

Codex 按下面结构创建或重构项目：

```text
go1_lewm_mpc/
├── README.md
├── TASKS.md
├── pyproject.toml
├── requirements.txt
├── configs/
│   ├── env/
│   │   └── go1_lewm_rough.yaml
│   ├── lewm/
│   │   ├── dummy_lewm.yaml
│   │   └── train_lewm.yaml
│   ├── mpc/
│   │   └── osqp_foothold.yaml
│   └── eval/
│       └── benchmark.yaml
├── go1_lewm_mpc/
│   ├── __init__.py
│   ├── common/
│   │   ├── types.py
│   │   ├── constants.py
│   │   └── math_utils.py
│   ├── envs/
│   │   ├── go1_env_wrapper.py
│   │   ├── obs_adapter.py
│   │   └── payload_randomization.py
│   ├── data/
│   │   ├── dataset_schema.py
│   │   ├── hdf5_writer.py
│   │   └── replay_loader.py
│   ├── world_model/
│   │   ├── base.py
│   │   ├── dummy_lewm.py
│   │   ├── lewm_adapter.py
│   │   ├── terrain_head.py
│   │   └── state_head.py
│   ├── foothold/
│   │   ├── candidate_generator.py
│   │   ├── phase_estimator.py
│   │   ├── risk_map.py
│   │   └── selector.py
│   ├── mpc/
│   │   ├── osqp_foothold.py
│   │   ├── cost_terms.py
│   │   └── constraints.py
│   ├── controllers/
│   │   ├── low_level_policy_wrapper.py
│   │   ├── command_filter.py
│   │   └── cue_injection.py
│   ├── eval/
│   │   ├── metrics.py
│   │   ├── benchmark_payload.py
│   │   └── benchmark_terrain.py
│   └── tests/
│       ├── test_types.py
│       ├── test_obs_adapter.py
│       ├── test_candidate_generator.py
│       ├── test_osqp_foothold.py
│       ├── test_cue_injection.py
│       └── test_closed_loop_smoke.py
└── scripts/
    ├── run_baseline.py
    ├── collect_dataset.py
    ├── train_lewm.py
    ├── run_closed_loop.py
    ├── eval_closed_loop.py
    └── profile_mpc.py
```

---

## 2. 数据结构要求

所有模块必须通过明确 dataclass 通信，不允许用散乱 dict 在模块之间传递核心数据。

### 2.1 `ObsPacket`

文件：`go1_lewm_mpc/common/types.py`

```python
from dataclasses import dataclass
from typing import Optional
import numpy as np

@dataclass
class ObsPacket:
    """One control-step observation packet in SI units."""
    t: float

    # Base state, world frame unless otherwise stated.
    base_pos_w: np.ndarray        # shape [3], meters
    base_quat_wxyz: np.ndarray    # shape [4], wxyz
    base_lin_vel_w: np.ndarray    # shape [3], m/s
    base_ang_vel_w: np.ndarray    # shape [3], rad/s

    # Joint state.
    joint_pos: np.ndarray         # shape [12], rad
    joint_vel: np.ndarray         # shape [12], rad/s

    # Foot state.
    foot_pos_b: np.ndarray        # shape [4, 3], meters, body frame
    foot_pos_w: np.ndarray        # shape [4, 3], meters, world frame
    foot_contact: np.ndarray      # shape [4], bool or int8

    # Commands and terrain.
    cmd_vel: np.ndarray           # shape [3], [vx, vy, yaw_rate]
    height_scan: Optional[np.ndarray]  # shape [Nh] or [H, W]
    last_action: Optional[np.ndarray]  # shape [12]

    # Payload/domain randomization info.
    payload_mass: float = 0.0
    payload_com_b: Optional[np.ndarray] = None  # shape [3]
```

Foot order 固定为：

```python
FOOT_ORDER = ["FL", "FR", "RL", "RR"]
```

Codex 必须在 `constants.py` 中定义该顺序，所有模块统一使用。

### 2.2 `LatentPacket`

```python
@dataclass
class LatentPacket:
    t: float
    z: np.ndarray                 # shape [D]
    terrain_feat: np.ndarray      # shape [Dt]
    dyn_feat: np.ndarray          # shape [Dd]
    uncertainty: float
```

### 2.3 `FootholdCandidatePacket`

```python
@dataclass
class FootholdCandidatePacket:
    t: float
    swing_leg_id: int             # 0..3
    candidates_b: np.ndarray      # shape [K, 3], body frame, meters
    candidates_w: np.ndarray      # shape [K, 3], world frame, meters
    risk: np.ndarray              # shape [K], lower is safer
    reach_cost: np.ndarray        # shape [K]
    total_score: np.ndarray       # shape [K], lower is better
```

### 2.4 `MpcPlanPacket`

```python
@dataclass
class MpcPlanPacket:
    t: float
    selected_leg_id: int
    selected_foothold_b: np.ndarray   # shape [3]
    selected_foothold_w: np.ndarray   # shape [3]
    velocity_bias: np.ndarray         # shape [3], [dvx, dvy, dyaw]
    confidence: float
    debug: dict
```

### 2.5 `LowLevelCue`

```python
@dataclass
class LowLevelCue:
    cmd_vel_corrected: np.ndarray     # shape [3]
    foothold_hint_b: Optional[np.ndarray] = None  # shape [4, 3]
    risk_summary: Optional[np.ndarray] = None     # shape [4]
```

---

## 3. PR-0：项目骨架和配置系统

### 目标

建立最小可运行 Python 包结构，支持配置加载、日志、单元测试和脚本入口。

### 需要实现

文件：

- `pyproject.toml`
- `requirements.txt`
- `go1_lewm_mpc/common/types.py`
- `go1_lewm_mpc/common/constants.py`
- `go1_lewm_mpc/common/math_utils.py`
- `configs/env/go1_lewm_rough.yaml`
- `configs/mpc/osqp_foothold.yaml`
- `configs/lewm/dummy_lewm.yaml`

### `requirements.txt` 初始内容

```text
numpy
scipy
h5py
pyyaml
tqdm
matplotlib
osqp
pytest
```

如果项目环境已经由 Isaac Lab 管理，不要在 requirements 里重复安装 torch、isaaclab、rsl_rl。只保留项目自身需要的轻量依赖。

### 验收标准

- `python -m pytest go1_lewm_mpc/tests -q` 能运行。
- `from go1_lewm_mpc.common.types import ObsPacket` 不报错。
- 所有 dataclass 有 shape 注释。
- 所有单位统一使用 SI units。

---

## 4. PR-1：跑通官方 Go1 rough baseline

### 目标

先验证 Isaac Lab 中 Go1 rough locomotion baseline 可运行。这个 PR 不接 LEWM，不接 MPC。

### 需要实现

文件：

- `scripts/run_baseline.py`
- `go1_lewm_mpc/envs/go1_env_wrapper.py`
- `README.md` 中加入运行说明

### 功能要求

`run_baseline.py` 提供以下参数：

```bash
python scripts/run_baseline.py   --task Isaac-Velocity-Rough-Unitree-Go1-v0   --num_envs 64   --headless   --duration_sec 30
```

如果当前 Isaac Lab 环境不能直接通过普通 Python 启动，脚本需要清楚提示用户改用：

```bash
./isaaclab.sh -p scripts/run_baseline.py --task Isaac-Velocity-Rough-Unitree-Go1-v0 --headless
```

### `Go1EnvWrapper` 接口

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

### 验收标准

- 可以启动 Isaac Lab Go1 rough 环境。
- 能 reset 和 step 至少 100 步。
- 不要求训练新 policy。
- 不要求渲染视频。
- 出错时输出明确错误信息，例如 Isaac Lab 未安装、task 未注册、缺少 rsl_rl。

---

## 5. PR-2：Observation Adapter

### 目标

把 Isaac Lab 原始观测整理成 `ObsPacket`。这是后续 LEWM、candidate generator 和 MPC 的共同输入。

### 需要实现

文件：

- `go1_lewm_mpc/envs/obs_adapter.py`
- `go1_lewm_mpc/tests/test_obs_adapter.py`

### 核心接口

```python
class ObsAdapter:
    def __init__(self, foot_order=("FL", "FR", "RL", "RR")):
        ...

    def from_isaac(self, raw_obs, env, env_id: int = 0) -> ObsPacket:
        """
        Convert Isaac Lab raw obs / scene tensors into one ObsPacket.
        """
        ...
```

### 最低要求

即使暂时不能从 Isaac Lab 拿到全部真实字段，也要实现 fallback：

- 如果 `foot_pos_b` 暂时拿不到，用零数组并打 warning。
- 如果 `height_scan` 暂时拿不到，用 `None`。
- 如果 `payload_mass` 暂时拿不到，用 `0.0`。
- 不允许静默返回错误 shape。

### 单元测试要求

`test_obs_adapter.py` 构造 fake raw obs，检查：

- `base_pos_w.shape == (3,)`
- `base_quat_wxyz.shape == (4,)`
- `joint_pos.shape == (12,)`
- `foot_pos_b.shape == (4, 3)`
- `cmd_vel.shape == (3,)`
- 所有 dtype 可以转换成 `np.float32`

### 验收标准

- `pytest go1_lewm_mpc/tests/test_obs_adapter.py -q` 通过。
- Adapter 可以在没有 Isaac Lab 的情况下用 fake data 测试。
- Adapter 在真实 Isaac Lab 环境中至少能输出一条 `ObsPacket` 并打印 summary。

---

## 6. PR-3：Dataset Collector

### 目标

从 baseline Go1 policy 运行中采集 LEWM 所需的离线数据。第一阶段先采集 HDF5，不训练。

### 需要实现

文件：

- `scripts/collect_dataset.py`
- `go1_lewm_mpc/data/dataset_schema.py`
- `go1_lewm_mpc/data/hdf5_writer.py`
- `go1_lewm_mpc/tests/test_dataset_schema.py`

### HDF5 schema

每个 episode 一个 group：

```text
/episode_000000/
    t                  [T]
    base_pos_w          [T, 3]
    base_quat_wxyz      [T, 4]
    base_lin_vel_w      [T, 3]
    base_ang_vel_w      [T, 3]
    joint_pos           [T, 12]
    joint_vel           [T, 12]
    foot_pos_b          [T, 4, 3]
    foot_pos_w          [T, 4, 3]
    foot_contact        [T, 4]
    cmd_vel             [T, 3]
    height_scan         [T, Nh] or [T, H, W]
    last_action         [T, 12]
    payload_mass        [T, 1]
    success             scalar
    fall                scalar
```

### 脚本接口

```bash
python scripts/collect_dataset.py   --task Isaac-Velocity-Rough-Unitree-Go1-v0   --num_envs 128   --episodes 200   --episode_len 500   --out data/go1_rough_payload_v0.hdf5   --headless
```

### 实现细节

- 支持 `--max_steps_per_file`，避免单个 HDF5 太大。
- 每 10 个 episode 打印一次统计。
- 必须记录 fall / termination 信息。
- 如果无法判断 fall，先用 base height threshold 作为 fallback。
- 默认先不保存 RGB pixels，避免数据过大；预留 `--save_pixels`。

### 验收标准

- Fake data 写入/读取测试通过。
- 真实 Isaac Lab 下能采集至少 5 个 episode。
- HDF5 可以被 `h5py.File(path, "r")` 正常打开。
- 数据 shape 与 schema 一致。

---

## 7. PR-4：DummyLEWM 和 LEWM 接口

### 目标

先不接真实 LEWM，先用 dummy 风险图支持完整闭环。之后真实 LEWM 只需要替换 `WorldModelBase` 的实现。

### 需要实现

文件：

- `go1_lewm_mpc/world_model/base.py`
- `go1_lewm_mpc/world_model/dummy_lewm.py`
- `go1_lewm_mpc/world_model/terrain_head.py`
- `go1_lewm_mpc/world_model/state_head.py`
- `go1_lewm_mpc/tests/test_lewm_heads.py`

### 抽象接口

```python
from abc import ABC, abstractmethod

class WorldModelBase(ABC):
    @abstractmethod
    def encode(self, obs: ObsPacket) -> LatentPacket:
        ...

    @abstractmethod
    def predict_risk(self, obs: ObsPacket, query_points_b: np.ndarray) -> np.ndarray:
        """
        Args:
            obs: current observation
            query_points_b: [K, 3] candidate footholds in body frame
        Returns:
            risk: [K], lower is safer
        """
        ...

    @abstractmethod
    def predict_state(self, obs: ObsPacket, horizon: int, dt: float) -> np.ndarray:
        """
        Returns:
            pred_state: [H, Nx]
        """
        ...
```

### `DummyLEWM` 行为

`DummyLEWM` 用可解释规则生成风险：

- 离 body 太远：高风险。
- 落脚点 z 明显低于/高于当前估计地面：高风险。
- 如果 height_scan 存在，则根据局部 roughness 增加风险。
- 如果 payload_mass 较大，则扩大保守区域。
- 中心安全区域风险低。

### 验收标准

- 不依赖 torch 也能运行。
- `predict_risk()` 输入 `[K, 3]`，输出 `[K]`。
- 所有 risk 是 finite number。
- 这个模块可以独立用 fake ObsPacket 测试。

---

## 8. PR-5：Heuristic Foothold Candidate Generator

### 目标

根据当前速度命令、腿相位、足端当前位置和安全边界，生成候选落脚点集合。

### 需要实现

文件：

- `go1_lewm_mpc/foothold/phase_estimator.py`
- `go1_lewm_mpc/foothold/candidate_generator.py`
- `go1_lewm_mpc/foothold/risk_map.py`
- `go1_lewm_mpc/tests/test_candidate_generator.py`

### 相位估计

第一阶段不要实现复杂 gait scheduler。用 contact state + last contact transition 估计 swing leg：

```python
class PhaseEstimator:
    def update(self, obs: ObsPacket) -> int:
        """
        Return swing_leg_id in {0,1,2,3}.
        If uncertain, choose the leg with lowest recent contact confidence.
        """
```

如果 contact 信息不可用，使用固定 trot 顺序 fallback：

```text
FL/RR -> FR/RL -> FL/RR -> ...
```

### 候选点生成

```python
class FootholdCandidateGenerator:
    def __init__(self, n_candidates_per_leg: int, max_step_x: float, max_step_y: float):
        ...

    def generate(self, obs: ObsPacket, swing_leg_id: int) -> np.ndarray:
        """
        Returns candidates_b: [K, 3]
        """
```

候选点生成规则：

- 以当前 swing leg 的 nominal foot position 为中心。
- 根据 `cmd_vel[0]` 向前偏移。
- 根据 `cmd_vel[1]` 横向偏移。
- 根据 `cmd_vel[2]` 添加 yaw 方向修正。
- 在椭圆可达域内采样 K 个点。
- 默认 z 从 height_scan 或当前足端高度估计；没有高度信息则 z=当前足端 z。

### 建议参数

文件：`configs/mpc/osqp_foothold.yaml`

```yaml
dt: 0.02
n_candidates_per_leg: 16
max_step_x: 0.18
max_step_y: 0.12
max_step_z: 0.10
nominal_stance:
  FL: [0.20,  0.12, -0.30]
  FR: [0.20, -0.12, -0.30]
  RL: [-0.20,  0.12, -0.30]
  RR: [-0.20, -0.12, -0.30]
```

### 验收标准

- 每次输出 shape `[K, 3]`。
- 候选点不超过 `max_step_x/max_step_y`。
- 前进命令增大时，候选点平均 x 应前移。
- 侧向命令增大时，候选点平均 y 应相应偏移。
- 测试覆盖 4 条腿。

---

## 9. PR-6：OSQP Foothold Selector

### 目标

实现最小可用 OSQP 选择器：从 K 个候选落脚点中选出一个最优点。第一阶段可以把离散选择近似为连续加权选择，或者先用 heuristic argmin，再保留 OSQP 接口。

### 需要实现

文件：

- `go1_lewm_mpc/mpc/osqp_foothold.py`
- `go1_lewm_mpc/mpc/cost_terms.py`
- `go1_lewm_mpc/mpc/constraints.py`
- `go1_lewm_mpc/foothold/selector.py`
- `go1_lewm_mpc/tests/test_osqp_foothold.py`

### 推荐第一版实现

为了避免混合整数优化，第一版不要做 binary selection。使用连续变量：

```text
u = [dx, dy]
```

优化目标：

```text
min
    w_risk       * interpolated_risk(dx, dy)
  + w_reach      * reachability_cost(dx, dy)
  + w_nominal    * ||[dx,dy] - nominal_step||^2
  + w_payload    * payload_margin_cost
```

如果风险插值太复杂，第一版可以这样做：

1. 对候选点计算总成本。
2. 取 top-M 候选点。
3. OSQP 在 top-M 的局部凸包内求一个连续点。
4. 最终投影到最近候选点。

### 接口

```python
class OSQPFootholdSelector:
    def __init__(self, cfg: dict):
        ...

    def select(
        self,
        obs: ObsPacket,
        swing_leg_id: int,
        candidates_b: np.ndarray,
        risk: np.ndarray,
    ) -> MpcPlanPacket:
        ...
```

### fallback 行为

如果 OSQP failed：

- 不要让系统崩溃。
- fallback 到 `np.argmin(total_score)`。
- 在 `MpcPlanPacket.debug["solver_status"]` 中记录失败原因。
- `confidence` 降低到 0.2 以下。

### 验收标准

- 输入 fake candidates + risk，可以输出 finite foothold。
- 高风险候选点不会被选中。
- 如果所有点风险相同，选择接近 nominal 的点。
- OSQP solve time 被记录。
- OSQP 失败时 fallback 正常。

---

## 10. PR-7：Cue Injection 到低层 policy

### 目标

把选出的 foothold plan 转换成低层 Go1 policy 能消费的信号。方案二第一阶段优先使用 velocity correction，不强行修改 policy 网络。

### 需要实现

文件：

- `go1_lewm_mpc/controllers/cue_injection.py`
- `go1_lewm_mpc/controllers/command_filter.py`
- `go1_lewm_mpc/controllers/low_level_policy_wrapper.py`
- `go1_lewm_mpc/tests/test_cue_injection.py`

### 第一阶段策略：velocity bias

把 selected foothold 相对 nominal foothold 的偏差转换为短时速度修正：

```python
def foothold_to_velocity_bias(
    obs: ObsPacket,
    plan: MpcPlanPacket,
    gain_xy: float,
    gain_yaw: float,
    max_bias: np.ndarray,
) -> np.ndarray:
    """
    Return [dvx, dvy, dyaw].
    """
```

建议规则：

```text
dx = selected_foothold_b[0] - nominal_foothold_b[0]
dy = selected_foothold_b[1] - nominal_foothold_b[1]

dvx = gain_xy * dx
dvy = gain_xy * dy
dyaw = gain_yaw * dy or based on left/right asymmetry
```

并做 clip：

```yaml
cue:
  gain_xy: 1.0
  gain_yaw: 0.5
  max_bias: [0.15, 0.10, 0.25]
  smoothing_alpha: 0.8
```

### 命令滤波

```python
class CommandFilter:
    def __init__(self, alpha: float, max_delta: np.ndarray):
        ...

    def update(self, cmd: np.ndarray) -> np.ndarray:
        ...
```

避免 corrected command 抖动过大，导致低层 policy 不稳定。

### `LowLevelPolicyWrapper`

```python
class LowLevelPolicyWrapper:
    def __init__(self, policy, use_cue: bool = True):
        ...

    def compute_action(self, raw_obs, cue: LowLevelCue):
        """
        If low-level policy accepts only cmd_vel, modify cmd_vel before policy inference.
        If observation augmentation is enabled, append cue fields to observation.
        """
        ...
```

第一版只要求支持 cmd_vel 修正。如果当前官方 policy 的 command buffer 不容易外部改写，则先在 wrapper 层保存 corrected command，并把它传给环境 command manager 或自定义 obs adapter。

### 验收标准

- `cmd_vel_corrected = cmd_vel + velocity_bias`。
- corrected command 被 clip 到安全范围。
- command filter 不产生 NaN。
- cue disabled 时，输出与 baseline 一致。
- fake policy 单元测试通过。

---

## 11. PR-8：Closed-loop Runner

### 目标

把 ObsAdapter、DummyLEWM、CandidateGenerator、OSQPSelector、CueInjection、LowLevelPolicyWrapper 串起来，跑第一个完整闭环。

### 需要实现

文件：

- `scripts/run_closed_loop.py`
- `go1_lewm_mpc/eval/metrics.py`
- `go1_lewm_mpc/tests/test_closed_loop_smoke.py`

### 脚本接口

```bash
python scripts/run_closed_loop.py   --task Isaac-Velocity-Rough-Unitree-Go1-v0   --num_envs 16   --duration_sec 60   --world_model dummy   --use_mpc true   --use_cue true   --headless
```

### 闭环伪代码

```python
env = Go1EnvWrapper(...)
obs_adapter = ObsAdapter()
wm = DummyLEWM(...)
phase = PhaseEstimator()
generator = FootholdCandidateGenerator(...)
selector = OSQPFootholdSelector(...)
cue_injector = CueInjector(...)
low_level = LowLevelPolicyWrapper(...)

raw_obs = env.reset()

while not done:
    obs = obs_adapter.from_isaac(raw_obs, env)

    swing_leg = phase.update(obs)
    candidates_b = generator.generate(obs, swing_leg)
    risk = wm.predict_risk(obs, candidates_b)

    plan = selector.select(obs, swing_leg, candidates_b, risk)
    cue = cue_injector.make_cue(obs, plan)

    action = low_level.compute_action(raw_obs, cue)
    raw_obs, reward, done, info = env.step(action)

    metrics.update(obs, plan, cue, info)
```

### 运行模式

必须支持三种模式：

```bash
# 纯 baseline
--use_mpc false --use_cue false

# 有候选点和风险评估，但不影响低层
--use_mpc true --use_cue false

# 完整方案二闭环
--use_mpc true --use_cue true
```

### 验收标准

- dummy world model 模式下可跑 10 秒不崩溃。
- baseline / no-cue / cue 三种模式都能运行。
- 日志中记录 selected foothold、risk、velocity bias、fall、base height。
- 出现 NaN 立刻停止并保存 debug dump。
- smoke test 不要求运动性能提升，只要求闭环稳定运行。

---

## 12. PR-9：Evaluation Scripts

### 目标

建立可重复实验，比较 baseline 与方案二。

### 需要实现

文件：

- `scripts/eval_closed_loop.py`
- `go1_lewm_mpc/eval/benchmark_payload.py`
- `go1_lewm_mpc/eval/benchmark_terrain.py`
- `go1_lewm_mpc/eval/metrics.py`

### 评测场景

至少支持：

```text
flat + 0 kg
rough + 0 kg
rough + 1 kg
rough + 2 kg
rough + random push + 1 kg
stepping stones + 1 kg
stairs + 1 kg
```

如果某些 terrain 暂时没有配置，先用 roughness level 代替，但要在日志中写清楚。

### 指标

必须记录：

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

第一阶段 slip proxy 可以用：

```text
foot_contact == True 且 foot velocity tangential norm > threshold
```

如果足端速度暂时拿不到，就先记录 `slip_proxy = NaN`，不要伪造结果。

### 输出格式

每次 eval 输出：

```text
runs/
└── YYYYMMDD_HHMMSS_scheme2_eval/
    ├── config.yaml
    ├── metrics.csv
    ├── summary.json
    ├── debug_selected_footholds.npz
    └── plots/
        ├── success_rate.png
        ├── fall_rate.png
        ├── mpc_solve_time.png
        └── risk_vs_success.png
```

### 验收标准

- 同一 config 可以重复运行。
- `summary.json` 中包含 git commit hash，如无法获取则写 `"unknown"`。
- baseline 与 cue 模式结果分开保存。
- `metrics.csv` 每一行对应一个 episode。

---

## 13. PR-10：真实 LEWM Adapter

### 目标

在 dummy 闭环稳定后，接入真实 LeWM backbone。此 PR 不负责把 LEWM 训练到最好，只负责完成接口。

### 需要实现

文件：

- `go1_lewm_mpc/world_model/lewm_adapter.py`
- `scripts/train_lewm.py`
- `configs/lewm/train_lewm.yaml`

### 接口要求

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

### 最低实现

如果真实 LeWM 暂时只有 latent rollout，没有 foothold risk head，则先实现：

```text
height_scan / local roughness / proprio features
    ↓
small MLP risk head
```

也就是说：

- LeWM backbone 提供 latent。
- `terrain_head.py` 接 query footholds。
- 输出每个 candidate 的 risk。
- 不要求端到端最优。

### 训练数据标签

第一版 risk label 可以用规则生成：

```text
risk = 1 if:
    foot slipped after contact
    or foot collision detected
    or body pitch/roll increased sharply after contact
    or episode terminated within short horizon
else:
    risk = 0
```

如果真实标签难以对齐，先做 pairwise ranking：

```text
same timestep 下，成功支撑区域 score 更低，失败/高扰动区域 score 更高
```

### 验收标准

- `LEWMAdapter` 可替换 `DummyLEWM`，不需要改 closed-loop runner。
- 如果 checkpoint 缺失，给出明确错误。
- 如果 GPU 不可用，支持 CPU fallback，但打印 warning。
- 输出 risk shape `[K]`，且 finite。
- 支持 `--world_model lewm --checkpoint path/to.ckpt`。

---

## 14. PR-11：Ablation

### 目标

验证方案二是否真的带来增益，而不是只增加复杂度。

### 需要实现

在 `scripts/eval_closed_loop.py` 中支持：

```bash
--ablation baseline
--ablation heuristic_only
--ablation dummy_lewm_risk
--ablation lewm_risk
--ablation lewm_risk_no_payload
--ablation lewm_risk_no_height
```

### Ablation 定义

| 模式 | 含义 |
|---|---|
| `baseline` | 官方 Go1 low-level policy，不加 MPC，不加 cue |
| `heuristic_only` | 候选点生成 + heuristic selector，但不用 LEWM risk |
| `dummy_lewm_risk` | 使用 DummyLEWM 风险 |
| `lewm_risk` | 使用真实 LEWM risk |
| `lewm_risk_no_payload` | LEWM 输入去掉 payload |
| `lewm_risk_no_height` | LEWM 输入去掉 height_scan |

### 验收标准

- 每个 ablation 模式可独立运行。
- 输出统一 metrics。
- 如果某模式未实现，脚本必须明确报 `NotImplementedError`，不要静默退化成 baseline。
- 生成一个 `ablation_summary.csv`。

---

## 15. 配置文件模板

### 15.1 `configs/env/go1_lewm_rough.yaml`

```yaml
task_name: Isaac-Velocity-Rough-Unitree-Go1-v0
num_envs: 64
headless: true
control_dt: 0.02

command:
  vx_range: [-0.6, 0.8]
  vy_range: [-0.3, 0.3]
  yaw_rate_range: [-0.6, 0.6]

payload:
  enabled: true
  mass_range_kg: [0.0, 2.0]
  com_range_b_m:
    x: [-0.05, 0.05]
    y: [-0.03, 0.03]
    z: [0.00, 0.08]

logging:
  save_height_scan: true
  save_pixels: false
  episode_len: 500
```

### 15.2 `configs/mpc/osqp_foothold.yaml`

```yaml
dt: 0.02
horizon: 5
n_candidates_per_leg: 16

candidate:
  max_step_x: 0.18
  max_step_y: 0.12
  max_step_z: 0.10
  nominal_stance:
    FL: [0.20,  0.12, -0.30]
    FR: [0.20, -0.12, -0.30]
    RL: [-0.20,  0.12, -0.30]
    RR: [-0.20, -0.12, -0.30]

weights:
  risk: 6.0
  reach: 4.0
  nominal: 2.0
  payload_margin: 2.0
  smoothness: 0.5

constraints:
  friction_coeff: 0.6
  payload_margin_scale: 1.25
  max_selected_step_x: 0.20
  max_selected_step_y: 0.14

solver:
  eps_abs: 1.0e-3
  eps_rel: 1.0e-3
  max_iter: 100
  polish: false
  warm_start: true
```

### 15.3 `configs/lewm/dummy_lewm.yaml`

```yaml
latent_dim: 192
risk:
  safe_radius_x: 0.16
  safe_radius_y: 0.10
  far_penalty: 5.0
  roughness_penalty: 3.0
  payload_penalty_scale: 1.5
  z_penalty: 2.0
```

### 15.4 `configs/eval/benchmark.yaml`

```yaml
episodes_per_case: 20
episode_len: 500

cases:
  - name: flat_0kg
    terrain: flat
    payload_mass: 0.0
  - name: rough_0kg
    terrain: rough
    payload_mass: 0.0
  - name: rough_1kg
    terrain: rough
    payload_mass: 1.0
  - name: rough_2kg
    terrain: rough
    payload_mass: 2.0
  - name: rough_push_1kg
    terrain: rough
    payload_mass: 1.0
    push: true

modes:
  - baseline
  - heuristic_only
  - dummy_lewm_risk
  - lewm_risk
```

---

## 16. Codex 执行顺序

Codex 必须按以下顺序执行，不要跳步：

```text
1. PR-0 项目骨架
2. PR-1 Go1 baseline wrapper
3. PR-2 ObsAdapter
4. PR-3 Dataset collector
5. PR-4 DummyLEWM
6. PR-5 Candidate generator
7. PR-6 OSQP selector
8. PR-7 Cue injection
9. PR-8 Closed-loop runner
10. PR-9 Evaluation scripts
11. PR-10 LEWMAdapter
12. PR-11 Ablation
```

每完成一个 PR，必须：

- 运行对应单元测试。
- 更新 README 中的运行命令。
- 在 `runs/` 或 `logs/` 中保存一个最小输出样例。
- 不要把大文件、checkpoint、HDF5 数据集提交到 git。
- 大文件路径写入 `.gitignore`。

---

## 17. 总体验收标准

项目第一阶段完成时，应满足：

### 功能

- 可运行官方 Go1 rough baseline。
- 可采集 Go1 locomotion dataset。
- 可在 dummy world model 下跑完整 scheme2 closed-loop。
- 可通过 OSQP 或 fallback selector 选择 candidate foothold。
- 可把 foothold result 转成 low-level velocity cue。
- 可比较 baseline / no-cue / cue 三种模式。

### 稳定性

- 10 秒 closed-loop smoke test 不崩溃。
- 无 NaN action。
- OSQP failed 时 fallback 正常。
- cue disabled 时行为接近 baseline。
- 所有核心数据 shape 有单元测试保护。

### 实验

- 至少完成 `rough + 0kg` 和 `rough + 1kg` 两个 case。
- 至少输出 success/fall/solve-time/risk 指标。
- 至少保存一次 `summary.json` 和 `metrics.csv`。
- 能为后续真实 LEWM 接入提供 dataset 和接口。

---

## 18. 给 Codex 的首个 Prompt

可以直接把下面这段作为 Codex 的第一条任务：

```text
You are modifying the repository for a Unitree Go1 + Isaac Lab + LEWM + MPC foothold-cue project.

Implement PR-0 and PR-1 only.

Goal:
Create the Python package skeleton, dataclass-based interfaces, config files, and a minimal Isaac Lab Go1 rough baseline runner. Do not implement LEWM, MPC, candidate generation, or cue injection yet.

Required files:
- pyproject.toml
- requirements.txt
- README.md
- configs/env/go1_lewm_rough.yaml
- configs/mpc/osqp_foothold.yaml
- configs/lewm/dummy_lewm.yaml
- go1_lewm_mpc/common/types.py
- go1_lewm_mpc/common/constants.py
- go1_lewm_mpc/common/math_utils.py
- go1_lewm_mpc/envs/go1_env_wrapper.py
- scripts/run_baseline.py
- go1_lewm_mpc/tests/test_types.py

Key requirements:
1. Define ObsPacket, LatentPacket, FootholdCandidatePacket, MpcPlanPacket, and LowLevelCue as dataclasses.
2. Define FOOT_ORDER = ["FL", "FR", "RL", "RR"].
3. Implement a Go1EnvWrapper with reset(), step(), get_raw_obs(), and close().
4. Implement scripts/run_baseline.py with CLI arguments:
   --task, --num_envs, --headless, --duration_sec.
5. If Isaac Lab is unavailable, fail gracefully with a clear installation/task error message.
6. Add pytest tests for dataclass construction and shape assumptions.
7. Do not add large data files or checkpoints.
8. Update README with the baseline run command.

Acceptance:
- python -m pytest go1_lewm_mpc/tests -q passes.
- python scripts/run_baseline.py --help works.
- The code is structured so later PRs can add ObsAdapter, DummyLEWM, OSQP, and cue injection without rewriting PR-0/PR-1.
```

---

## 19. 关键提醒

方案二的重点不是一开始把 footstep 完全控制住，而是先验证：

```text
LEWM / risk prediction 是否能提供有用的 terrain-aware cue
```

所以第一阶段成功标准不是：

```text
每一步都精准踩到 MPC 指定点
```

而是：

```text
在 rough terrain + payload 下，相比 baseline，fall rate 降低、stumble/slip proxy 降低、速度跟踪不要明显恶化。
```

只要这个闭环跑通，后续再升级为方案一，即：

```text
LEWM foothold candidates → MPC hard footstep plan → swing trajectory / WBC tracking
```

会更稳，也更容易写成研究贡献。

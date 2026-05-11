# LeWM 完整功能训练手册

这份手册对应当前根目录项目，不修改 `工程文件/` 里的历史版本。目标是把 LeWM 训练成 MPC 可用的地形/状态预测层，而不是训练成 12D 关节动作控制器。

目标闭环是：

```text
Go1 local heightmap / observation frame
  -> LeWM encoder -> z_t
  -> LeWM predictor(z_t, MidAction) -> future latent
  -> latent_cost / uncertainty / auxiliary probes
  -> MPC foothold/body planner
  -> controller
```

## 1. 当前 LeWM 学什么

当前本地 LeWM checkpoint 的核心语义是：

```text
frame[t] -> encoder -> z[t]
z[t] + action[t] -> predictor -> predicted z[t+1]
frame[t+1] -> encoder -> target z[t+1]
loss = MSE(predicted z[t+1], target z[t+1]) + optional sigreg
```

默认数据形状：

```text
frame:      [T, 1, 64, 64]
action:     [T, 13]
next_frame: [T, 1, 64, 64]
done:       [T]
```

13D `MidAction` 不是 12D joint action，布局是：

```text
[vx, vy, yaw_rate,
 dvx, dvy, dyaw,
 selected_leg_onehot(4),
 foothold_delta_xyz]
```

其中：

- `command` action mode 只填前三维命令速度，其余维度为 0。
- `touchdown` action mode 会从观测到的下一步触地事件推断 `selected_leg_onehot` 和 `foothold_delta_xyz`。这是 hindsight 预训练标签，不等价于真实 MPC 计划日志。

## 2. 数据应该包含什么

最低可训练数据：

```text
t
base_pos_w / base_quat_wxyz
base_lin_vel_w / base_ang_vel_w
joint_pos / joint_vel
foot_pos_b / foot_pos_w
foot_contact
cmd_vel
height_scan
last_action
payload_mass
success / fall
```

完整功能路线还应该逐步补：

```text
terrain_context / support_map
MPC selected_leg_id
MPC selected_foothold_b/w
candidate footholds
candidate latent_cost / risk / support score
base roll/pitch/slip/contact labels
failure and near-failure episodes
```

如果没有真实 MPC 计划日志，先用 `--action_mode touchdown` 做足端条件预训练；等闭环能记录 MPC 计划后，再把 converter 改成读取真实 `selected_leg_id + selected_foothold`。

## 3. 一键脚本

入口：

```bash
.venv/bin/python scripts/train_full_lewm_pipeline.py --help
```

先只打印计划，不启动 Isaac：

```bash
.venv/bin/python scripts/train_full_lewm_pipeline.py \
  --terrain_preset rough \
  --run_name smoke_plan \
  --episodes 2 \
  --episode_len 100 \
  --action_mode touchdown \
  --epochs 1 \
  --batch_size 64 \
  --limit_batches 2 \
  --device cpu \
  --print_only
```

采集 rough 地形、转换、训练：

```bash
.venv/bin/python scripts/train_full_lewm_pipeline.py \
  --terrain_preset rough \
  --run_name rough_touchdown_v1 \
  --episodes 50 \
  --episode_len 500 \
  --num_envs 16 \
  --action_mode touchdown \
  --epochs 20 \
  --batch_size 256 \
  --device cuda
```

采集 sparse 复杂地形，包括 mixed sparse 和 plum piles：

```bash
.venv/bin/python scripts/train_full_lewm_pipeline.py \
  --terrain_preset sparse \
  --run_name sparse_touchdown_v1 \
  --episodes 50 \
  --episode_len 500 \
  --num_envs 16 \
  --action_mode touchdown \
  --epochs 30 \
  --batch_size 256 \
  --device cuda
```

采集 rough + sparse 全部地形：

```bash
.venv/bin/python scripts/train_full_lewm_pipeline.py \
  --terrain_preset full \
  --run_name full_touchdown_v1 \
  --episodes 80 \
  --episode_len 500 \
  --num_envs 16 \
  --action_mode touchdown \
  --epochs 40 \
  --batch_size 256 \
  --device cuda
```

输出默认在：

```text
runs/lewm_full/<run_name>/
  data/raw_<terrain>.hdf5
  data/lewm_sequences_<terrain>_<action_mode>.hdf5
  data/lewm_sequences_merged_<action_mode>.hdf5
  checkpoints/lewm_full_<run_name>_<action_mode>.ckpt
  pipeline_manifest.json
```

## 4. 使用已有 raw 数据重新转换训练

如果已经有 raw HDF5，不想重新采集：

```bash
.venv/bin/python scripts/train_full_lewm_pipeline.py \
  --terrains rough \
  --skip_collect \
  --raw_path data/go1_rough_baseline_policy_heightscan_50x500.hdf5 \
  --out_dir runs/lewm_full/retrain_from_raw \
  --action_mode touchdown \
  --epochs 20 \
  --batch_size 256 \
  --device cuda
```

如果已经有 LeWM sequence HDF5，只训练：

```bash
.venv/bin/python scripts/train_full_lewm_pipeline.py \
  --terrains rough \
  --skip_collect \
  --skip_convert \
  --sequence_path data/go1_rough_lewm_sequences_500x500.hdf5 \
  --checkpoint_out checkpoints/lewm_retrain_from_sequences.ckpt \
  --epochs 20 \
  --batch_size 256 \
  --device cuda
```

## 5. 单独三段命令

采集：

```bash
/home/kaga/IsaacLab/isaaclab.sh -p scripts/collect_dataset.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 16 \
  --episodes 50 \
  --episode_len 500 \
  --out data/raw_rough.hdf5 \
  --headless \
  --policy_checkpoint .pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go1-v0/exported/policy.pt
```

转换：

```bash
.venv/bin/python scripts/convert_rollout_to_lewm_dataset.py \
  --in data/raw_rough.hdf5 \
  --out data/lewm_sequences_rough_touchdown.hdf5 \
  --only_success \
  --frame_size 64 64 \
  --action_mode touchdown
```

训练：

```bash
.venv/bin/python scripts/train_lewm.py \
  --config configs/lewm/train_lewm.yaml \
  --dataset data/lewm_sequences_rough_touchdown.hdf5 \
  --out checkpoints/lewm_rough_touchdown_v1.ckpt \
  --epochs 20 \
  --batch_size 256 \
  --device cuda
```

## 6. 训练后检查

检查 checkpoint 元数据：

```bash
.venv/bin/python - <<'PY'
import torch
path = "runs/lewm_full/rough_touchdown_v1/checkpoints/lewm_full_rough_touchdown_v1_touchdown.ckpt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
for key in ["format", "frame_shape", "action_dim", "latent_dim", "sequence_length", "final_metrics"]:
    print(key, ckpt.get(key))
PY
```

检查 sequence 里的 action 是否真的有 foothold 条件：

```bash
.venv/bin/python - <<'PY'
import h5py
import numpy as np
path = "runs/lewm_full/rough_touchdown_v1/data/lewm_sequences_rough_touchdown.hdf5"
with h5py.File(path, "r") as f:
    legs = []
    deltas = []
    for ep in sorted(k for k in f if k.startswith("episode_")):
        action = f[ep]["world_model"]["action"][()]
        legs.append(action[:, 6:10].sum(axis=1))
        deltas.append(np.linalg.norm(action[:, 10:13], axis=1))
    legs = np.concatenate(legs)
    deltas = np.concatenate(deltas)
    print("foothold_label_steps:", int(np.count_nonzero(legs > 0.5)))
    print("nonzero_delta_steps:", int(np.count_nonzero(deltas > 1e-6)))
PY
```

## 7. 当前限制

当前脚本能让 LeWM 进入可训练、可复现的完整数据流水线，但还不是最终论文级闭环：

- `touchdown` 是从观测反推的足端条件，不是真实 MPC action log。
- 当前 `train_lewm.py` 只训练 encoder + latent predictor，没有训练 traversability/support/risk auxiliary heads。
- 当前复杂地形采集仍依赖官方 Go1 policy 是否能在该地形上走出有效轨迹。
- 如果要服务 `MPC -> foothold/body plan -> IK/controller -> 12D action`，后续必须记录真实 MPC 计划，并训练 LeWM 对候选 action rollout 的 latent cost 和不确定性排序能力。

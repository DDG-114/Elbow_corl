# Go1 Rough Dataset 到 LeWM 训练使用指南

这份指南记录当前项目里从 Isaac Lab 采集 baseline 数据、检查数据、转换成 LeWM sequence dataset、再训练本地 LeWM checkpoint 的完整流程。

## 0. 前提

所有命令默认在项目根目录执行：

```bash
cd /home/kaga/corl2026
```

当前链路使用官方 Go1 rough locomotion policy 作为低层策略，只采集它在 Isaac Lab rough terrain 环境里的 rollout。LeWM 只学习：

```text
height_scan / local frame -> latent z_t
latent z_t + 13D MidAction -> predicted latent z_{t+1}
```

LeWM 不输出 12D joint action，也不替换官方低层策略。

## 1. 采集 Raw Rollout 数据

推荐先采小规模 pilot，确认 `height_scan` 非空：

```bash
TERM=xterm VIRTUAL_ENV=/home/kaga/corl2026/.venv \
/home/kaga/IsaacLab/isaaclab.sh -p scripts/collect_dataset.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 1 \
  --episodes 50 \
  --episode_len 1000 \
  --out data/go1_rough_baseline_policy_heightscan_pilot.hdf5 \
  --headless \
  --policy_checkpoint .pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go1-v0/exported/policy.pt
```

确认 pilot 正常后，再采训练用数据。例如当前已经验证过的 50 条 episode：

```bash
TERM=xterm VIRTUAL_ENV=/home/kaga/corl2026/.venv \
/home/kaga/IsaacLab/isaaclab.sh -p scripts/collect_dataset.py \
  --task Isaac-Velocity-Rough-Unitree-Go1-v0 \
  --num_envs 1 \
  --episodes 50 \
  --episode_len 500 \
  --out data/go1_rough_baseline_policy_heightscan_50x500.hdf5 \
  --headless \
  --policy_checkpoint .pretrained_checkpoints/rsl_rl/Isaac-Velocity-Rough-Unitree-Go1-v0/exported/policy.pt
```

关键参数：

- `--episodes`：最多采多少条 episode。
- `--episode_len`：每条 episode 最多走多少个 env step。
- `--policy_checkpoint`：官方 Go1 policy，不提供时会采 zero-action 数据，不适合训练当前 LeWM。
- `--headless`：无 GUI 采集，避开 Isaac Sim GUI/RTX 崩溃风险。

## 2. 检查 Raw HDF5

采集后先检查 episode 数量、长度、success/fall 和 `height_scan`：

```bash
.venv/bin/python - <<'PY'
import h5py
import numpy as np
from pathlib import Path

path = Path("data/go1_rough_baseline_policy_heightscan_50x500.hdf5")
with h5py.File(path, "r") as f:
    episodes = sorted(k for k in f.keys() if k.startswith("episode_"))
    lengths = [f[ep]["t"].shape[0] for ep in episodes]
    successes = [bool(f[ep]["success"][()]) for ep in episodes]
    falls = [bool(f[ep]["fall"][()]) for ep in episodes]
    first = episodes[0]
    height = f[first]["height_scan"][()]

    print("episodes:", len(episodes))
    print("length min/max:", min(lengths), max(lengths))
    print("full-length episodes:", sum(length == 500 for length in lengths))
    print("successes:", sum(successes))
    print("falls:", sum(falls))
    print("height shape first:", height.shape)
    print("height nonempty:", height.shape[1:] != (0,))
    print("height finite:", bool(np.isfinite(height).all()))
PY
```

可用数据的基本要求：

```text
height_scan 不是 (T, 0)
height_scan finite=True
有足够 success=True 的 episode
最好先使用 full-length 且 success=True 的 episode
```

## 3. 转换成 LeWM Sequence Dataset

Raw rollout 不能直接给 LeWM sequence dataset 用，需要转换出 `world_model` group：

```bash
rm -f data/go1_rough_lewm_sequences_heightscan_50x500.hdf5

.venv/bin/python scripts/convert_rollout_to_lewm_dataset.py \
  --in data/go1_rough_baseline_policy_heightscan_50x500.hdf5 \
  --out data/go1_rough_lewm_sequences_heightscan_50x500.hdf5 \
  --only_success \
  --require_full_length \
  --expected_length 500 \
  --frame_size 64 64
```

转换后每个 episode 会包含：

```text
episode_xxxxxx/world_model/frame       [T, 1, 64, 64]
episode_xxxxxx/world_model/action      [T, 13]
episode_xxxxxx/world_model/next_frame  [T, 1, 64, 64]
episode_xxxxxx/world_model/done        [T]
episode_xxxxxx/world_model/probe/...   auxiliary labels
```

其中 `action` 是 13D MidAction：

```text
[vx, vy, yaw_rate, dvx, dvy, dyaw, selected_leg_onehot(4), foothold_delta_xyz]
```

当前从 baseline rollout 转换时只使用 command-only MidAction，所以后 10 维通常为 0。

## 4. 检查 LeWM Dataset

转换后检查 schema 和训练窗口数量：

```bash
.venv/bin/python - <<'PY'
import h5py
import numpy as np
from go1_lewm_mpc.data.lewm_sequence_dataset import LeWMSequenceDataset

path = "data/go1_rough_lewm_sequences_heightscan_50x500.hdf5"
with h5py.File(path, "r") as f:
    episodes = sorted(k for k in f.keys() if k.startswith("episode_"))
    first = episodes[0]
    wm = f[first]["world_model"]
    print("episodes:", len(episodes))
    print("first:", first)
    print("frame:", wm["frame"].shape, wm["frame"].dtype, bool(np.isfinite(wm["frame"][()]).all()))
    print("action:", wm["action"].shape, wm["action"].dtype, bool(np.isfinite(wm["action"][()]).all()))
    print("next_frame:", wm["next_frame"].shape, wm["next_frame"].dtype)
    print("done:", wm["done"].shape, bool(wm["done"][-1]))

dataset = LeWMSequenceDataset(path, seq_len=8)
print("seq_len_8_windows:", len(dataset))
PY
```

当前 50x500 数据的参考结果：

```text
episodes: 47
frame: (500, 1, 64, 64)
action: (500, 13)
seq_len_8_windows: 23171
```

## 5. 训练前 Dry Run

先跑 dry-run，确认配置和路径解析正常：

```bash
.venv/bin/python scripts/train_lewm.py \
  --config configs/lewm/train_lewm.yaml \
  --dataset data/go1_rough_lewm_sequences_heightscan_50x500.hdf5 \
  --out checkpoints/lewm_heightscan_dryrun.ckpt \
  --dry_run
```

`--dry_run` 不会写真实训练 checkpoint，只会打印 plan 和一轮 synthetic loss。

## 6. Smoke 训练

正式训练前先用少量 batch 跑通：

```bash
.venv/bin/python scripts/train_lewm.py \
  --config configs/lewm/train_lewm.yaml \
  --dataset data/go1_rough_lewm_sequences_heightscan_50x500.hdf5 \
  --out checkpoints/lewm_heightscan_smoke.ckpt \
  --epochs 1 \
  --batch_size 64 \
  --limit_batches 2 \
  --device cpu
```

成功时会看到：

```text
LEWM training complete:
  checkpoint: checkpoints/lewm_heightscan_smoke.ckpt
```

然后验证 checkpoint 可加载：

```bash
.venv/bin/python - <<'PY'
import numpy as np
import torch
from go1_lewm_mpc.tests.fixtures import make_fake_height_scan, make_fake_obs_packet
from go1_lewm_mpc.world_model.action_adapter import MID_ACTION_VECTOR_DIM
from go1_lewm_mpc.world_model.lewm_adapter import LEWMAdapter

path = "checkpoints/lewm_heightscan_smoke.ckpt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("format:", ckpt["format"])
print("final_metrics:", ckpt["final_metrics"])

model = LEWMAdapter(path, cfg={}, device="cpu")
obs = make_fake_obs_packet(height_scan=make_fake_height_scan(rough=True))
rollout = model.rollout_latent(obs, np.zeros((3, MID_ACTION_VECTOR_DIM), dtype=np.float32), dt=0.02)
print("rollout_len:", len(rollout))
print("latent_shape:", rollout[0].z.shape)
print("finite:", bool(np.isfinite(np.stack([x.z for x in rollout])).all()))
PY
```

## 7. 正式训练

使用 GPU 训练当前本地 LeWM：

```bash
.venv/bin/python scripts/train_lewm.py \
  --config configs/lewm/train_lewm.yaml \
  --dataset data/go1_rough_lewm_sequences_heightscan_50x500.hdf5 \
  --out checkpoints/lewm_heightscan_v0.ckpt \
  --epochs 20 \
  --batch_size 2048 \
  --device cuda
```

如果显存不够，降低 batch size：

```bash
--batch_size 512
```

训练日志里的指标：

- `prediction`：一步 latent 预测 MSE，越低表示 `z_t + action_t -> z_{t+1}` 越准。
- `sigreg`：latent 防塌缩正则，越低通常表示 latent 维度更有变化。
- `total`：`prediction + lambda_sigreg * sigreg`。

注意：目前脚本保存最后一轮 checkpoint，还没有 validation split 和 best checkpoint 保存。

## 8. 训练后检查 Checkpoint

```bash
.venv/bin/python - <<'PY'
import torch

path = "checkpoints/lewm_heightscan_v0.ckpt"
ckpt = torch.load(path, map_location="cpu", weights_only=False)
print("format:", ckpt["format"])
print("frame_shape:", ckpt["frame_shape"])
print("action_dim:", ckpt["action_dim"])
print("latent_dim:", ckpt["latent_dim"])
print("batches_seen:", ckpt["batches_seen"])
print("final_metrics:", ckpt["final_metrics"])
PY
```

## 9. 跑测试

代码改动或训练脚本调整后，至少跑：

```bash
.venv/bin/python -m pytest go1_lewm_mpc/tests -q
```

数据、checkpoint、HDF5、日志、视频都在 `.gitignore` 中，不应该提交。

## 10. 下一步

训练出 `checkpoints/lewm_heightscan_v0.ckpt` 后，下一步不是继续盲目采更多数据，而是把 checkpoint 接入 closed-loop：

```text
run_closed_loop.py
  --world_model local_lewm
  --planner_mode local_lewm_latent_cost
  --world_model_checkpoint checkpoints/lewm_heightscan_v0.ckpt
```

目标是验证 trained latent rollout 是否能通过 latent cost 影响 foothold selector，而不是替换低层 Go1 policy。

# LeWM + MPC 迭代闭环实现计划

这份计划记录当前项目推荐的工程路线：先把操控闭环跑通，再离线训练 LeWM，把训练好的 LeWM 冻结后接入 MPC，最后通过新闭环继续采集数据并迭代。

## 1. 核心判断

LeWM 和 MPC 之间不是互相等待的死循环。实际规划时，MPC 可以先提出一批候选 high-level actions，然后 LeWM 评估这些候选动作的未来 latent 代价，MPC 再选择动作：

```text
obs / frame
  -> LeWM encode -> z_t

candidate generator / MPC proposal:
  action_1, action_2, ..., action_K

LeWM rollout:
  action_i -> future latent_i -> latent_cost_i / uncertainty_i

MPC:
  terrain/support constraints + latent_cost_i
  -> selected foothold/body plan

controller:
  plan -> 12D action
```

因此训练路线应当是离线迭代，而不是一开始就在控制循环中实时更新 LeWM。

## 2. 不推荐一开始在线边控边训

在线同时训练 LeWM 和执行 MPC 的风险很高：

```text
LeWM 参数变化 -> MPC cost surface 变化
MPC 行为变化 -> 数据分布变化
controller 未稳定 -> 大量失败/异常数据
失败数据污染训练 -> 闭环进一步不稳定
```

第一版应保持：

```text
控制时 LeWM 冻结
训练时离线更新 LeWM
更新 checkpoint 后再进入下一轮闭环
```

## 3. 阶段 A：先准备可运行操控闭环

目标是先有一个不依赖学习世界模型也能执行的基础控制链：

```text
terrain/support heuristic
  -> candidate footholds
  -> foothold/body plan
  -> IK/controller 或官方 policy cue
  -> 12D action / Isaac action
```

第一版可以先用保守策略：

```text
TerrainContext / support_map
  -> terrain-aware candidate generator
  -> heuristic / OSQP selector
  -> official policy cue 或简单 IK controller
```

这一阶段要记录完整日志，而不是追求 LeWM 参与决策。

## 4. 阶段 B：采集带 high-level action 的数据

每个 timestep 至少记录：

```text
frame_t
MidAction_t
frame_{t+1}
cmd_vel
selected_leg_id
selected_foothold_b
selected_foothold_w
candidate footholds
terrain_context / support_map metadata
base state
foot_pos_b / foot_pos_w
foot_contact 或可替代 touchdown 标签
success / fall
```

如果暂时没有真实 MPC plan 日志，可以用 hindsight touchdown 作为预训练标签；但最终要替换成真实的：

```text
MPC selected_leg_id + selected_foothold_b/w
```

## 5. 阶段 C：离线训练 LeWM

训练目标保持 LeWM-style world model 语义：

```text
frame_t -> encoder -> z_t
z_t + MidAction_t -> predictor -> predicted z_{t+1}
frame_{t+1} -> encoder -> target z_{t+1}
```

主损失：

```text
prediction loss in latent space
regularization loss
```

辅助 probe 可以在后续加入：

```text
risk / traversability
support score
base height / roll / pitch
slip / contact consistency
```

这些 probe 只能辅助 MPC，不能替代核心世界模型目标。

## 6. 阶段 D：冻结 LeWM 后接入 MPC

部署时使用固定 checkpoint：

```text
current frame -> z_t
candidate MidActions [K, 13]
  -> LeWM rollout
  -> latent_cost [K]
  -> uncertainty [K]
```

MPC 选择时综合：

```text
terrain support cost
reachability cost
body stability cost
latent_cost
uncertainty penalty
```

这一阶段先做 debug/eval，不直接承诺复杂地形稳定通过。

## 7. 阶段 E：闭环采集再训练

当 `LeWM + MPC` 能跑起来后，用它采集第二轮数据：

```text
policy_0 / heuristic_mpc -> dataset_0 -> lewm_0
lewm_0 + mpc -> dataset_1 -> lewm_1
lewm_1 + mpc -> dataset_2 -> lewm_2
```

每轮都保留：

```text
checkpoint
dataset manifest
terrain distribution
success/fall metrics
latent_cost debug dump
```

## 8. 当前项目的直接下一步

当前已有 command-only LeWM checkpoint 可以先用于 smoke：

```text
local_lewm checkpoint
planner_mode=latent_cost
candidate scoring
debug selected foothold and cost distribution
```

但要进入真正 foothold-conditioned LeWM，必须先修数据标签：

```text
当前旧 raw 数据 foot_contact 全 False
touchdown 标签无法生成
selected_leg_onehot / foothold_delta 仍为 0
```

下一步优先级：

```text
1. 修 collector / ObsAdapter 的 foot_contact 来源，或实现 kinematic touchdown 标签。
2. 让 closed-loop 显式接入 terrain-aware generator/selector。
3. 用真实 MPC plan 日志训练 foothold-conditioned LeWM。
4. 冻结新 LeWM checkpoint 后再进入 MPC 复杂地形验证。
```

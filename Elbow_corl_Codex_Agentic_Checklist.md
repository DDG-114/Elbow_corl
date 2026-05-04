# Elbow_corl — Codex Agentic Coding Checklist

## 目标

把当前 `Elbow_corl` 仓库重构为更贴近 `lucas-maes/le-wm` 原始语义的项目。

当前重点不是重写低层控制器，而是修正世界模型职责：

```text
LeWM core:
  encode
  encode_frame
  predict_next_latent
  rollout_latent

Auxiliary probes:
  predict_risk
  predict_state
  terrain_head
  payload_head
```

## 最重要的修改方向

1. `predict_risk()` 不再是世界模型核心。
2. 添加 `WorldModelInputFrame`，把 Go1 height_scan 转成 LeWM-style frame。
3. 添加 `MidAction`，作为 LeWM predictor 的 action conditioning。
4. 添加 `rollout_latent()`，用于 latent planning。
5. 添加 `UpstreamLeWMBridge`，先 mock，再真实接入 `lucas-maes/le-wm`。
6. 让 MPC cost 后续能使用 latent rollout cost，而不只用 risk。
7. evaluation ablation 区分：
   - auxiliary risk
   - latent rollout cost
   - upstream mock
   - local LeWM

## 推荐执行顺序

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

## 每个 PR 的固定要求

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

## 第一条 Codex 指令

```text
Read README.md, TASKS.md, AGENTS.md, CODEX_TASK_QUEUE.md, and Elbow_corl_Codex_Agentic_Checklist.md.

Implement PR-00 only.

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

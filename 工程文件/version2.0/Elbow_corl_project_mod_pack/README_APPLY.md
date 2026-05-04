# Elbow_corl Project Modification Pack

这个文件夹用于把当前 `Elbow_corl` 仓库改造成更贴近 `lucas-maes/le-wm` 原始设计的 Go1 + Isaac Lab + LeWM + MPC 项目。

## 使用方式

假设当前目录结构为：

```text
Elbow_corl/
Elbow_corl_project_mod_pack/
```

推荐先不要直接覆盖代码，而是：

```bash
cd Elbow_corl
git checkout -b lewm-alignment-refactor
```

然后复制 overlay 文件：

```bash
cp ../Elbow_corl_project_mod_pack/repo_overlay/AGENTS.md .
cp ../Elbow_corl_project_mod_pack/repo_overlay/CODEX_TASK_QUEUE.md .
cp ../Elbow_corl_project_mod_pack/repo_overlay/CODEX_PROMPTS.md .
cp ../Elbow_corl_project_mod_pack/repo_overlay/.gitignore.additions .gitignore.additions
mkdir -p docs
cp ../Elbow_corl_project_mod_pack/repo_overlay/docs/*.md docs/
```

如果你愿意把 `.gitignore.additions` 合并进 `.gitignore`：

```bash
cat .gitignore.additions >> .gitignore
```

然后把 `codex_specs/Elbow_corl_Codex_Agentic_Checklist.md` 放到仓库根目录：

```bash
cp ../Elbow_corl_project_mod_pack/codex_specs/Elbow_corl_Codex_Agentic_Checklist.md .
```

## 推荐第一步

先给 Codex 这个 prompt：

```text
Read README.md, TASKS.md, AGENTS.md, and Elbow_corl_Codex_Agentic_Checklist.md.

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

## 设计重点

本次修改不是让 LeWM 直接输出控制动作，而是让项目世界模型职责更接近原始 LeWM：

```text
observation frame
    ↓
encoder
    ↓
latent z_t
    ↓ + high-level action / MidAction
predictor
    ↓
future latent sequence
    ↓
latent planning score / auxiliary probes
    ↓
MPC / cue injection
    ↓
existing Go1 low-level locomotion policy
```

所以：

```text
LeWM core = encode + predict_next_latent + rollout_latent
risk / terrain / state prediction = auxiliary probe heads
control = MPC / cue injection / low-level policy
```

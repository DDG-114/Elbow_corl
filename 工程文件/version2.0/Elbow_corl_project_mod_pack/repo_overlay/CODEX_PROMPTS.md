# CODEX_PROMPTS.md

Copy one prompt at a time into Codex. Do not ask Codex to implement the entire queue at once.

---

## Prompt PR-00

```text
Read README.md, TASKS.md, AGENTS.md, and CODEX_TASK_QUEUE.md.

Implement PR-00 only.

This is an existing repository, not a greenfield project.
Do not rewrite working Python modules.
Only normalize root-level agentic documentation and task status.

Allowed files:
AGENTS.md
CODEX_TASK_QUEUE.md
CODEX_PROMPTS.md
README.md
.gitignore

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

## Prompt PR-01

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-01 only: remove runtime dependency on tests fixtures.

Move FakeIsaacEnv into go1_lewm_mpc/mock/fake_isaac_env.py and update runtime scripts to import from the runtime-safe mock module.

Do not modify:
- world_model semantics
- mpc
- controllers
- ObsPacket schema

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

## Prompt PR-02

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-02 only: add WorldModelInputFrame contract.

Goal:
Add a conversion layer that turns ObsPacket.height_scan into a LeWM-style [C,H,W] observation frame.

Requirements:
- Support height_scan None, [Nh], and [H,W].
- Output [1,64,64] by default.
- No torch dependency.
- No Isaac Lab dependency.

Run:
python -m pytest go1_lewm_mpc/tests/test_world_model_input_frame.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## Prompt PR-03

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-03 only: refactor WorldModelBase semantics toward LeWM-style latent dynamics.

Add core methods:
- encode_frame
- predict_next_latent
- rollout_latent

Keep predict_risk and predict_state, but mark them as auxiliary probe methods in docstrings.

DummyLEWM must implement all methods without torch.
Do not implement real upstream le-wm loading yet.

Run:
python -m pytest go1_lewm_mpc/tests/test_lewm_semantic_interface.py -q
python -m pytest go1_lewm_mpc/tests/test_dummy_lewm.py -q

Final response:
Files changed:
Tests run:
Assumptions:
Limitations:
Next recommended task:
Stop.
```

---

## Prompt PR-04

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-04 only: add MidAction and action adapter for LeWM action conditioning.

Rules:
- MidAction is high-level.
- Do not use 12D joint actions as LeWM action.
- Do not modify the low-level locomotion policy.

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

## Prompt PR-05

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-05 only: add world_model/factory.py.

Add backend choices:
- dummy
- local_lewm
- upstream_lewm_mock

Do not implement real upstream le-wm loading yet.
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

## Prompt PR-06

```text
Read AGENTS.md and CODEX_TASK_QUEUE.md.

Implement PR-06 only: add UpstreamLeWMBridge skeleton.

This PR does not need to run real lucas-maes/le-wm.
It must define a clean boundary and mock mode.
No upstream import at module import time.
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

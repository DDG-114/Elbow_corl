# Agentic Coding Workflow for Go1 + LEWM + MPC

This file explains how to use Codex effectively for this project.

## Do Not Use One Giant Prompt

Do not send:

```text
Read TASKS.md and implement everything.
```

This project depends on Isaac Lab and simulator-specific behavior. A giant prompt will likely lead to unstable architecture and untestable code.

Use this workflow instead:

```text
TASKS.md              = master design document
AGENTS.md            = coding-agent behavior rules
CODEX_TASK_QUEUE.md  = small executable task list
CODEX_PROMPTS.md     = copy-paste prompts for each task
docs/ENVIRONMENT.md  = local Isaac Lab setup and simulator caveats
tests/fixtures.py    = mock-first test base
```

## Recommended Loop

For each task:

```text
1. Ask Codex to implement one task only.
2. Let Codex edit files.
3. Run the tests it reports.
4. Review the diff manually.
5. Commit.
6. Move to the next task.
```

## Best First Three Tasks

Start with:

```text
Task 001 — Project Skeleton
Task 002 — Baseline Go1 Wrapper
Task 003 — ObsAdapter
```

Do not start with LEWM or MPC before the observation and baseline wrapper exist.

## Local vs Cloud Codex

Cloud Codex may be useful for:

```text
dataclasses
config files
mock tests
HDF5 schema
DummyLEWM
candidate generator
OSQP selector
cue injection
metrics
documentation
```

Local Codex / local terminal is better for:

```text
Isaac Sim
Isaac Lab
GPU simulation
task registry
rendering
RSL-RL policy loading
real rollout collection
```

## Commit Strategy

Use one commit per task:

```bash
git checkout -b task-001-skeleton
# Codex implements Task 001
python -m pytest go1_lewm_mpc/tests -q
git add .
git commit -m "Add project skeleton and core dataclasses"
```

Then:

```bash
git checkout -b task-002-baseline-wrapper
# Codex implements Task 002
python -m pytest go1_lewm_mpc/tests -q
git commit -m "Add Isaac Lab Go1 baseline wrapper"
```

## Human Review Checklist

Before accepting a Codex change, check:

- Did it implement only the requested task?
- Did it import Isaac Lab in unit-testable modules?
- Did it touch future modules unnecessarily?
- Did it add large files?
- Did tests pass without Isaac Lab?
- Did simulator-specific failures have clear messages?
- Did it preserve the low-level control boundary?
- Did it avoid direct 12D action output from LEWM?

## Suggested Branches

```text
task-001-skeleton
task-002-baseline-wrapper
task-003-obs-adapter
task-004-dataset-collector
task-005-dummy-lewm
task-006-candidate-generator
task-007-osqp-selector
task-008-cue-injection
task-009-closed-loop
task-010-evaluation
task-011-lewm-adapter
task-012-ablation
```

## What Counts as Progress

Early progress is not measured by robot walking better.

Early progress is:

```text
clean interfaces
mock tests
baseline runnable
dataset schema
dummy closed loop
safe fallbacks
repeatable evaluation
```

Only after these exist should you judge performance improvement.

## Core Architecture Reminder

The correct Phase 1 relationship is:

```text
LEWM predicts terrain/risk/state tendency
MPC selects safer foothold candidate
Cue injection modifies high-level command slightly
Low-level Go1 policy still handles gait and joint control
```

The wrong Phase 1 relationship is:

```text
LEWM directly outputs Go1 joint actions
```

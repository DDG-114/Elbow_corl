# CODEX_REVIEW_CHECKLIST.md

Use this checklist after every Codex task.

## Scope

- [ ] Did Codex implement only the requested task?
- [ ] Did it avoid implementing future tasks early?
- [ ] Did it avoid changing unrelated files?
- [ ] Did it preserve the architecture boundary in `AGENTS.md`?

## Dependency Safety

- [ ] Unit tests do not require Isaac Lab.
- [ ] Isaac Lab imports are lazy and isolated.
- [ ] Simulator-unavailable errors are clear.
- [ ] No hidden dependency on GPU for lightweight tests.

## Low-Level Control Boundary

- [ ] LEWM does not output 12D joint actions.
- [ ] Low-level policy is not rewritten.
- [ ] No WBC / OCS2 / Crocoddyl / torque-level NMPC added in Phase 1.
- [ ] Foothold plan is used as cue/bias, not hard joint control.

## Data Safety

- [ ] No HDF5/checkpoint/log/video committed.
- [ ] `.gitignore` covers generated artifacts.
- [ ] Test fixtures are small and deterministic.

## Tests

- [ ] Focused tests passed.
- [ ] Full lightweight test suite passed.
- [ ] Tests include invalid shape/error cases where relevant.
- [ ] Mock tests work without Isaac Lab.

## Code Quality

- [ ] Core data uses dataclasses.
- [ ] Shapes and units are documented.
- [ ] No circular imports.
- [ ] No silent fallback to baseline for unimplemented requested modes.
- [ ] NaN/invalid values are checked in safety-critical code.

## Reporting

- [ ] Codex reported files changed.
- [ ] Codex reported tests run.
- [ ] Codex reported assumptions.
- [ ] Codex reported limitations.
- [ ] Codex recommended next task.

"""Wrapper for passing optional low-level cues to a policy object."""

from __future__ import annotations

from typing import Any

from go1_lewm_mpc.common.types import LowLevelCue


class LowLevelPolicyWrapper:
    """Call a low-level policy with optional cue information."""

    def __init__(self, policy: Any, use_cue: bool = True):
        self.policy = policy
        self.use_cue = bool(use_cue)
        self.last_corrected_command = None

    def compute_action(self, raw_obs, cue: LowLevelCue | None = None):
        """Compute an action while preserving baseline behavior when cue is disabled."""
        if self.use_cue and cue is not None:
            self.last_corrected_command = cue.cmd_vel_corrected.copy()
            if hasattr(self.policy, "compute_action"):
                return self.policy.compute_action(raw_obs, cue=cue)
            if callable(self.policy):
                try:
                    return self.policy(raw_obs, cue=cue)
                except TypeError:
                    return self.policy(raw_obs)

        self.last_corrected_command = None
        if hasattr(self.policy, "compute_action"):
            return self.policy.compute_action(raw_obs, cue=None)
        if callable(self.policy):
            return self.policy(raw_obs)
        raise TypeError("policy must be callable or expose compute_action(raw_obs, cue=None)")

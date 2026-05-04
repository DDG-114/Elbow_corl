"""Simple contact-based swing-leg phase estimator."""

from __future__ import annotations

import numpy as np

from go1_lewm_mpc.common.constants import N_FEET
from go1_lewm_mpc.common.types import ObsPacket


class PhaseEstimator:
    """Estimate the current swing leg without a full gait scheduler."""

    def __init__(self):
        self._fallback_order = (0, 3, 1, 2)
        self._fallback_index = 0
        self._last_contact: np.ndarray | None = None

    def update(self, obs: ObsPacket) -> int:
        """Return a swing leg id in {0, 1, 2, 3}."""
        contact = np.asarray(obs.foot_contact, dtype=bool)
        if contact.shape != (N_FEET,):
            return self._next_fallback()

        non_contact = np.flatnonzero(~contact)
        if non_contact.size > 0:
            leg_id = int(non_contact[0])
            self._last_contact = contact
            return leg_id

        if self._last_contact is not None:
            touchdown = np.flatnonzero((~self._last_contact) & contact)
            if touchdown.size > 0:
                self._last_contact = contact
                return int((touchdown[0] + 1) % N_FEET)

        self._last_contact = contact
        return self._next_fallback()

    def _next_fallback(self) -> int:
        leg_id = self._fallback_order[self._fallback_index % len(self._fallback_order)]
        self._fallback_index += 1
        return int(leg_id)

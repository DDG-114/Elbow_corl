"""Payload randomization bridge for fake and Isaac-like environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PayloadSpec:
    """Payload mass and body-frame center of mass offset."""

    mass_kg: float
    com_b: np.ndarray

    def __post_init__(self) -> None:
        mass = float(self.mass_kg)
        com = np.asarray(self.com_b, dtype=np.float32)
        if mass < 0.0 or not np.isfinite(mass):
            raise ValueError(f"mass_kg must be finite and non-negative, got {self.mass_kg}")
        if com.shape != (3,):
            raise ValueError(f"com_b must have shape (3,), got {com.shape}")
        if not np.all(np.isfinite(com)):
            raise ValueError("com_b must contain only finite values")
        object.__setattr__(self, "mass_kg", mass)
        object.__setattr__(self, "com_b", com)

    def as_metadata(self) -> dict:
        return {
            "payload_mass": float(self.mass_kg),
            "payload_com_b": self.com_b.astype(np.float32).copy(),
        }


@dataclass
class PayloadRandomizer:
    """Sample and apply payload specs.

    Fake environments should expose ``set_payload_spec`` or mutable metadata.
    Real Isaac Lab environments must provide an explicit payload hook; otherwise
    ``apply`` raises instead of pretending that physics changed.
    """

    mass_range_kg: tuple[float, float] = (0.0, 2.0)
    com_range_b: tuple[tuple[float, float, float], tuple[float, float, float]] = (
        (-0.02, -0.02, 0.02),
        (0.02, 0.02, 0.08),
    )

    def __post_init__(self) -> None:
        low_mass, high_mass = (float(self.mass_range_kg[0]), float(self.mass_range_kg[1]))
        if low_mass < 0.0 or high_mass < low_mass:
            raise ValueError(f"mass_range_kg must be ordered and non-negative, got {self.mass_range_kg}")
        low_com = _as_vec3(self.com_range_b[0], "com_range_b low")
        high_com = _as_vec3(self.com_range_b[1], "com_range_b high")
        if np.any(high_com < low_com):
            raise ValueError("com_range_b high values must be >= low values")
        self.mass_range_kg = (low_mass, high_mass)
        self.com_range_b = (tuple(float(v) for v in low_com), tuple(float(v) for v in high_com))

    def sample(self, rng: np.random.Generator) -> PayloadSpec:
        """Sample one payload spec from configured uniform ranges."""
        if not isinstance(rng, np.random.Generator):
            raise ValueError("rng must be a numpy.random.Generator")
        mass = float(rng.uniform(self.mass_range_kg[0], self.mass_range_kg[1]))
        com = rng.uniform(np.asarray(self.com_range_b[0]), np.asarray(self.com_range_b[1])).astype(np.float32)
        return PayloadSpec(mass_kg=mass, com_b=com)

    def apply(self, env: Any, spec: PayloadSpec, env_ids=None) -> None:
        """Apply payload to fake metadata or explicit simulator hooks."""
        payload = _coerce_spec(spec)
        target = _unwrap_env(env)

        if hasattr(target, "set_payload_spec"):
            target.set_payload_spec(payload, env_ids=env_ids)
            return
        if hasattr(target, "payload_randomizer_apply"):
            target.payload_randomizer_apply(payload, env_ids=env_ids)
            return
        if hasattr(target, "payload_metadata"):
            target.payload_metadata = payload.as_metadata()
            return
        if isinstance(getattr(target, "metadata", None), dict):
            target.metadata.update(payload.as_metadata())
            return
        if _is_runtime_fake_env(target):
            target.payload_metadata = payload.as_metadata()
            return

        raise NotImplementedError(
            "PayloadRandomizer.apply requires a fake env metadata hook or an explicit Isaac Lab payload hook."
        )


def payload_spec_from_mapping(data: dict | None, default_mass_kg: float = 0.0) -> PayloadSpec:
    """Create a payload spec from scenario/config metadata."""
    source = data or {}
    mass = source.get("payload_mass", source.get("mass_kg", default_mass_kg))
    com = source.get("payload_com_b", source.get("com_b", np.zeros(3, dtype=np.float32)))
    return PayloadSpec(mass_kg=float(mass), com_b=np.asarray(com, dtype=np.float32))


def _coerce_spec(spec: PayloadSpec) -> PayloadSpec:
    if not isinstance(spec, PayloadSpec):
        raise ValueError(f"spec must be a PayloadSpec, got {type(spec).__name__}")
    return spec


def _as_vec3(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (3,):
        raise ValueError(f"{name} must have shape (3,), got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _unwrap_env(env: Any) -> Any:
    candidate = getattr(env, "env", env)
    return getattr(candidate, "unwrapped", candidate)


def _is_runtime_fake_env(env: Any) -> bool:
    return env.__class__.__module__.startswith("go1_lewm_mpc.mock.")

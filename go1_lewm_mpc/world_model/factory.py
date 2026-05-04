"""Factory for selectable world-model backends."""

from __future__ import annotations

from typing import Any

from go1_lewm_mpc.world_model.base import WorldModelBase
from go1_lewm_mpc.world_model.dummy_lewm import DummyLEWM, DummyLEWMConfig
from go1_lewm_mpc.world_model.lewm_adapter import LEWMAdapter
from go1_lewm_mpc.world_model.upstream_lewm_bridge import UpstreamLeWMBridge

WORLD_MODEL_BACKENDS = ("dummy", "local_lewm", "upstream_lewm_mock")


def build_world_model(
    backend: str,
    cfg: dict | None = None,
    checkpoint_path: str | None = None,
    device: str = "cpu",
) -> WorldModelBase:
    """Build a world model backend by name.

    Supported backends are ``dummy``, ``local_lewm``, and ``upstream_lewm_mock``.
    No real upstream lucas-maes/le-wm loading is performed in this PR.
    """

    backend_name = _normalize_backend(backend)
    cfg = dict(cfg or {})

    if backend_name == "dummy":
        if checkpoint_path:
            raise ValueError("backend='dummy' does not accept checkpoint_path")
        return DummyLEWM(_dummy_config_from_cfg(cfg))

    if backend_name == "local_lewm":
        resolved_checkpoint = checkpoint_path or cfg.get("checkpoint_path")
        if not resolved_checkpoint:
            raise ValueError("backend='local_lewm' requires checkpoint_path")
        return LEWMAdapter(str(resolved_checkpoint), cfg=cfg, device=device)

    if backend_name == "upstream_lewm_mock":
        if checkpoint_path or _requests_real_upstream_loading(cfg):
            raise NotImplementedError(
                "Real upstream lucas-maes/le-wm loading is not implemented yet. "
                "upstream_lewm_mock only supports mock mode without checkpoint_path/upstream_repo."
            )
        return UpstreamLeWMBridge(
            upstream_repo=None,
            checkpoint_path=None,
            cfg=cfg,
            device=device,
            allow_mock=True,
        )

    raise ValueError(f"unsupported world model backend: {backend!r}")


def _normalize_backend(backend: str) -> str:
    text = str(backend).strip().lower()
    if text not in WORLD_MODEL_BACKENDS:
        raise ValueError(f"backend must be one of {WORLD_MODEL_BACKENDS}, got {backend!r}")
    return text


def _requests_real_upstream_loading(cfg: dict) -> bool:
    if cfg.get("backend") in {"upstream_lewm", "real_upstream_lewm"}:
        return True
    if cfg.get("upstream_repo"):
        return True
    if cfg.get("checkpoint_path") and cfg.get("mock", True) is False:
        return True
    return False


def _dummy_config_from_cfg(cfg: dict[str, Any]) -> DummyLEWMConfig:
    allowed = set(DummyLEWMConfig.__dataclass_fields__)
    kwargs = {key: cfg[key] for key in allowed if key in cfg}
    return DummyLEWMConfig(**kwargs)

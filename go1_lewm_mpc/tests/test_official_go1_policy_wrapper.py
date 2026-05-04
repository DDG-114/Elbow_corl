import numpy as np
import pytest

from go1_lewm_mpc.controllers import OfficialGo1PolicyWrapper
from go1_lewm_mpc.common.types import LowLevelCue


def test_missing_official_policy_checkpoint_has_clear_error(tmp_path) -> None:
    missing = tmp_path / "missing_policy.pt"

    with pytest.raises(FileNotFoundError, match="Official Go1 policy checkpoint does not exist"):
        OfficialGo1PolicyWrapper(str(missing), device="cpu")


def test_official_policy_wrapper_runs_torchscript_policy(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "policy.pt"
    _save_policy(checkpoint, obs_dim=6, action_dim=12, torch=torch)
    wrapper = OfficialGo1PolicyWrapper(str(checkpoint), device="cpu")

    action = wrapper.compute_action({"policy": np.ones((1, 6), dtype=np.float32)})

    assert tuple(action.shape) == (1, 12)
    assert torch.isfinite(action).all()


def test_official_policy_wrapper_injects_cue_by_command_indices(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "policy.pt"
    _save_policy(checkpoint, obs_dim=6, action_dim=12, torch=torch)
    wrapper = OfficialGo1PolicyWrapper(
        str(checkpoint),
        device="cpu",
        command_indices=(0, 1, 2),
        strict_cue=True,
    )
    cue = LowLevelCue(cmd_vel_corrected=np.array([0.2, -0.1, 0.05], dtype=np.float32))

    action = wrapper.compute_action({"policy": np.zeros((1, 6), dtype=np.float32)}, cue=cue)

    assert tuple(action.shape) == (1, 12)
    assert np.allclose(wrapper.last_corrected_command, cue.cmd_vel_corrected)
    assert action[0, 0].item() != 0.0


def test_official_policy_wrapper_requires_cue_injection_when_strict(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "policy.pt"
    _save_policy(checkpoint, obs_dim=6, action_dim=12, torch=torch)
    wrapper = OfficialGo1PolicyWrapper(str(checkpoint), device="cpu", strict_cue=True)
    cue = LowLevelCue(cmd_vel_corrected=np.array([0.2, -0.1, 0.05], dtype=np.float32))

    with pytest.raises(RuntimeError, match="Could not inject low-level cue"):
        wrapper.compute_action({"policy": np.zeros((1, 6), dtype=np.float32)}, cue=cue)


def test_official_policy_wrapper_cpu_fallback_warns(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    checkpoint = tmp_path / "policy.pt"
    _save_policy(checkpoint, obs_dim=6, action_dim=12, torch=torch)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.warns(RuntimeWarning, match="falling back to CPU"):
        wrapper = OfficialGo1PolicyWrapper(str(checkpoint), device="cuda")

    assert wrapper.device.type == "cpu"


def _save_policy(path, obs_dim: int, action_dim: int, torch) -> None:
    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(obs_dim, action_dim)

        def forward(self, obs):
            return self.linear(obs)

    policy = Policy().eval()
    scripted = torch.jit.script(policy)
    scripted.save(str(path))

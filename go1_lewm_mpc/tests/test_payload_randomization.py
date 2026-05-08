import csv
import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from go1_lewm_mpc.envs.obs_adapter import ObsAdapter
from go1_lewm_mpc.envs.payload_randomization import PayloadRandomizer, PayloadSpec, payload_spec_from_mapping
from go1_lewm_mpc.mock.fake_isaac_env import FakeIsaacEnv

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_payload_spec_validates_and_serializes_metadata() -> None:
    spec = PayloadSpec(mass_kg=1.5, com_b=np.array([0.01, -0.02, 0.06], dtype=np.float32))

    metadata = spec.as_metadata()

    assert metadata["payload_mass"] == pytest.approx(1.5)
    assert np.allclose(metadata["payload_com_b"], [0.01, -0.02, 0.06])
    with pytest.raises(ValueError, match="mass_kg"):
        PayloadSpec(mass_kg=-1.0, com_b=np.zeros(3, dtype=np.float32))
    with pytest.raises(ValueError, match="com_b"):
        PayloadSpec(mass_kg=1.0, com_b=np.zeros(2, dtype=np.float32))


def test_payload_randomizer_samples_within_range() -> None:
    randomizer = PayloadRandomizer(
        mass_range_kg=(0.5, 2.0),
        com_range_b=((-0.01, -0.02, 0.03), (0.01, 0.02, 0.08)),
    )

    spec = randomizer.sample(np.random.default_rng(123))

    assert 0.5 <= spec.mass_kg <= 2.0
    assert np.all(spec.com_b >= np.array([-0.01, -0.02, 0.03], dtype=np.float32))
    assert np.all(spec.com_b <= np.array([0.01, 0.02, 0.08], dtype=np.float32))


def test_payload_randomizer_apply_records_fake_env_metadata_and_obs_adapter_uses_it() -> None:
    env = FakeIsaacEnv()
    spec = PayloadSpec(mass_kg=2.25, com_b=np.array([0.02, 0.0, 0.07], dtype=np.float32))

    PayloadRandomizer().apply(env, spec)
    packet = ObsAdapter().from_isaac(env.reset(), env=env)

    assert packet.payload_mass == pytest.approx(2.25)
    assert np.allclose(packet.payload_com_b, [0.02, 0.0, 0.07])


def test_payload_randomizer_does_not_pretend_real_apply_without_hook() -> None:
    class RealLikeEnv:
        pass

    with pytest.raises(NotImplementedError, match="PayloadRandomizer.apply"):
        PayloadRandomizer().apply(RealLikeEnv(), PayloadSpec(mass_kg=1.0, com_b=np.zeros(3, dtype=np.float32)))


def test_payload_spec_from_mapping_reads_scenario_payload() -> None:
    spec = payload_spec_from_mapping({"payload_mass": 2.0, "payload_com_b": [0.0, 0.01, 0.05]})

    assert spec.mass_kg == pytest.approx(2.0)
    assert np.allclose(spec.com_b, [0.0, 0.01, 0.05])


def test_eval_fake_writes_payload_metadata(tmp_path: Path) -> None:
    out_dir = tmp_path / "eval"
    cmd = [
        ".venv/bin/python",
        "scripts/eval_closed_loop.py",
        "--fake",
        "--episodes",
        "1",
        "--duration_sec",
        "0.04",
        "--out_dir",
        str(out_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True)

    assert result.returncode == 0, result.stderr
    with (out_dir / "metrics.csv").open("r", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    rough_2kg_rows = [row for row in rows if row["scenario"] == "rough_2kg"]
    assert rough_2kg_rows
    assert {row["payload_mass_kg"] for row in rough_2kg_rows} == {"2.0"}
    assert all("payload_com_b" in row for row in rough_2kg_rows)


def test_eval_real_mode_skips_zero_payload_apply_but_requires_nonzero_payload_hook(monkeypatch) -> None:
    module = _load_script("eval_closed_loop.py", "go1_eval_payload_test")
    plan = module._mode_plan("baseline", {})

    class MinimalWrapper:
        apply_calls = 0

        def __init__(self, **kwargs):
            pass

        def reset(self):
            return FakeIsaacEnv(episode_len=3).reset()

        def step(self, action=None):
            return FakeIsaacEnv(episode_len=3).step(action)

        def apply_payload(self, spec):
            type(self).apply_calls += 1
            raise NotImplementedError("payload hook missing")

        def close(self):
            pass

    def fake_run_closed_loop(**kwargs):
        return type("Metrics", (), {"episode_metrics": lambda self, **row_kwargs: {}})()

    monkeypatch.setattr(module, "Go1EnvWrapper", MinimalWrapper)
    monkeypatch.setattr(module, "_load_run_closed_loop", lambda: fake_run_closed_loop)

    module._run_episode({"duration_sec": 0.02}, {"name": "flat_0kg", "payload_mass": 0.0}, plan, fake=False)
    assert MinimalWrapper.apply_calls == 0

    with pytest.raises(NotImplementedError, match="payload hook missing"):
        module._run_episode({"duration_sec": 0.02}, {"name": "rough_2kg", "payload_mass": 2.0}, plan, fake=False)
    assert MinimalWrapper.apply_calls == 1


def test_collect_dataset_exposes_payload_cli_parser() -> None:
    module = _load_script("collect_dataset.py", "go1_collect_dataset_payload_test")

    assert module._parse_vec3("0.0,0.1,0.2", "--payload_com_b") == [0.0, 0.1, 0.2]
    with pytest.raises(ValueError, match="exactly three"):
        module._parse_vec3("0.0,0.1", "--payload_com_b")

    argv = [
        "collect_dataset.py",
        "--task",
        "Isaac-Velocity-Rough-Unitree-Go1-v0",
        "--reset_xy_range",
        "1.5",
        "--reset_yaw_range",
        "1.2",
        "--max_init_terrain_level",
        "3",
        "--headless",
    ]
    original = sys.argv[:]
    try:
        sys.argv = argv[:]
        module._strip_collector_args_from_kit_argv()
        assert sys.argv == ["collect_dataset.py"]
    finally:
        sys.argv = original


def test_collect_dataset_can_annotate_payload_metadata(tmp_path: Path) -> None:
    module = _load_script("collect_dataset.py", "go1_collect_dataset_payload_metadata_test")

    class Writer:
        def __init__(self):
            self._file = type("File", (), {"attrs": {}})()

    writer = Writer()
    spec = PayloadSpec(mass_kg=1.75, com_b=np.array([0.01, 0.02, 0.06], dtype=np.float32))

    module._annotate_payload_metadata(writer, spec)

    assert writer._file.attrs["payload_mass_kg"] == pytest.approx(1.75)
    assert np.allclose(writer._file.attrs["payload_com_b"], [0.01, 0.02, 0.06])


def test_collect_dataset_env_hook_overrides_reset_pose_and_terrain_level() -> None:
    module = _load_script("collect_dataset.py", "go1_collect_dataset_env_hook_test")

    class ResetBase:
        params = {
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
        }

    class Terrain:
        max_init_terrain_level = None

    class Scene:
        terrain = Terrain()

    class Events:
        reset_base = ResetBase()

    class EnvCfg:
        scene = Scene()
        events = Events()

    args = type(
        "Args",
        (),
        {
            "reset_xy_range": 1.75,
            "reset_yaw_range": 0.9,
            "max_init_terrain_level": 4,
        },
    )()

    env_cfg = EnvCfg()
    module._make_collection_env_hook(args)(env_cfg)

    pose_range = env_cfg.events.reset_base.params["pose_range"]
    assert pose_range["x"] == (-1.75, 1.75)
    assert pose_range["y"] == (-1.75, 1.75)
    assert pose_range["yaw"] == (-0.9, 0.9)
    assert env_cfg.scene.terrain.max_init_terrain_level == 4


def test_collect_dataset_done_env_indices_supports_vectorized_masks() -> None:
    module = _load_script("collect_dataset.py", "go1_collect_dataset_done_masks_test")

    terminated = np.array([False, True, False, False], dtype=bool)
    truncated = np.array([False, False, True, False], dtype=bool)

    done = module._done_env_indices(terminated, truncated)

    assert done == {1, 2}


def test_collect_dataset_select_env_value_handles_scalar_and_vector() -> None:
    module = _load_script("collect_dataset.py", "go1_collect_dataset_select_env_value_test")

    assert module._select_env_value(0.3, 0) == pytest.approx(0.3)
    assert module._select_env_value(np.array([0.1, 0.2, 0.3], dtype=np.float32), 2) == pytest.approx(0.3)
    with pytest.raises(IndexError, match="env_id"):
        module._select_env_value(np.array([0.1], dtype=np.float32), 3)


def test_collect_dataset_main_source_initializes_completed_episodes() -> None:
    path = REPO_ROOT / "scripts" / "collect_dataset.py"
    text = path.read_text(encoding="utf-8")

    assert "completed_episodes = 0" in text


def _load_script(script_name: str, module_name: str):
    path = REPO_ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module

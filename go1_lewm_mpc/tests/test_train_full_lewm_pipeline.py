from pathlib import Path
import importlib.util

import h5py


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "train_full_lewm_pipeline.py"
SPEC = importlib.util.spec_from_file_location("train_full_lewm_pipeline", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_full_lewm_pipeline = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_full_lewm_pipeline)
merge_sequence_files = train_full_lewm_pipeline.merge_sequence_files


def test_merge_sequence_files_keeps_all_episode_groups(tmp_path: Path) -> None:
    first = tmp_path / "first.hdf5"
    second = tmp_path / "second.hdf5"
    out = tmp_path / "merged.hdf5"
    _write_minimal_sequence_file(first, episode_count=2)
    _write_minimal_sequence_file(second, episode_count=1)

    merge_sequence_files(
        [first, second],
        out,
        terrain_names=["rough", "plum_piles"],
        action_mode="touchdown",
    )

    with h5py.File(out, "r") as file:
        episodes = sorted(name for name in file.keys() if name.startswith("episode_"))
        assert episodes == ["episode_000000", "episode_000001", "episode_000002"]
        assert file.attrs["action_mode"] == "touchdown"
        assert file.attrs["episode_count"] == 3
        assert file["episode_000000"].attrs["source_terrain"] == "rough"
        assert file["episode_000002"].attrs["source_terrain"] == "plum_piles"


def _write_minimal_sequence_file(path: Path, episode_count: int) -> None:
    with h5py.File(path, "w") as file:
        for index in range(episode_count):
            group = file.create_group(f"episode_{index:06d}")
            group.create_dataset("success", data=True)
            wm = group.create_group("world_model")
            wm.create_dataset("frame", shape=(2, 1, 4, 4), dtype="float32")
            wm.create_dataset("action", shape=(2, 13), dtype="float32")
            wm.create_dataset("next_frame", shape=(2, 1, 4, 4), dtype="float32")
            wm.create_dataset("done", data=[False, True])

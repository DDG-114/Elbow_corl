#!/usr/bin/env python3
"""Convert raw Go1 rollout HDF5 files into LeWM sequence HDF5 files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.data.lewm_converter import RolloutToLeWMConfig, convert_rollout_file_to_lewm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in", dest="input_path", required=True, help="Input raw rollout HDF5 path.")
    parser.add_argument("--out", dest="output_path", required=True, help="Output LeWM sequence HDF5 path.")
    parser.add_argument("--frame_size", nargs=2, type=int, default=(64, 64), metavar=("H", "W"))
    parser.add_argument("--no_normalize_frames", action="store_true", help="Disable per-frame heightmap normalization.")
    parser.add_argument("--only_success", action="store_true", help="Keep only success=True episodes.")
    parser.add_argument("--require_full_length", action="store_true", help="Keep only full-length episodes.")
    parser.add_argument("--expected_length", type=int, default=None, help="Expected full episode length.")
    parser.add_argument("--min_length", type=int, default=2, help="Minimum episode length to convert.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    cfg = RolloutToLeWMConfig(
        frame_size=(int(args.frame_size[0]), int(args.frame_size[1])),
        normalize_frames=not args.no_normalize_frames,
        only_success=bool(args.only_success),
        require_full_length=bool(args.require_full_length),
        expected_length=args.expected_length,
        min_length=int(args.min_length),
    )
    summary = convert_rollout_file_to_lewm(args.input_path, args.output_path, cfg)
    print("Rollout -> LeWM conversion complete:")
    print(f"  input: {summary.input_path}")
    print(f"  output: {summary.output_path}")
    print(f"  episodes_seen: {summary.episodes_seen}")
    print(f"  episodes_written: {summary.episodes_written}")
    print(f"  episodes_skipped: {summary.episodes_skipped}")
    print(f"  skipped_reasons: {summary.skipped_reasons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

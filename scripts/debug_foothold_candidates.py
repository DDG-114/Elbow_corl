#!/usr/bin/env python3
"""Debug terrain-aware foothold candidate generation in mock mode."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from go1_lewm_mpc.common.constants import N_FEET, N_JOINTS
from go1_lewm_mpc.common.types import ObsPacket
from go1_lewm_mpc.foothold.foothold_utils import body_points_to_world_xy, yaw_from_quat_wxyz
from go1_lewm_mpc.foothold.terrain_aware_generator import TerrainAwareFootholdCandidateGenerator
from go1_lewm_mpc.terrains.registry import make_terrain_generator
from go1_lewm_mpc.terrains.support_map import batch_query_support


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", default="beam", choices=["flat", "beam", "stepping_stones", "mixed"])
    parser.add_argument("--swing_leg", type=int, default=0, help="0 FL, 1 FR, 2 RL, 3 RR")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    terrain_gen = make_terrain_generator({"type": args.terrain})
    context = terrain_gen.query_context(np.array([0.0, 0.0, 0.32], dtype=np.float32), 0.0, rng)
    obs = _make_obs(context)
    generator = TerrainAwareFootholdCandidateGenerator()
    candidates = generator.generate(obs, args.swing_leg)
    yaw = yaw_from_quat_wxyz(obs.base_quat_wxyz)
    candidates_w_xy = body_points_to_world_xy(candidates[:, :2], obs.base_pos_w, yaw)
    support = None
    if context.support_map is not None:
        support = batch_query_support(context.support_map, candidates_w_xy, context.map_origin_w, context.map_resolution)

    print("terrain:", context.name)
    print("swing_leg:", args.swing_leg)
    print("candidate_count:", len(candidates))
    print("candidates_b:\n", candidates)
    if support is not None:
        print("support_values:", support)
        print("safe_count:", int(np.sum(support > 0.5)))


def _make_obs(terrain_context) -> ObsPacket:
    foot_pos_b = np.array(
        [
            [0.20, 0.12, -0.30],
            [0.20, -0.12, -0.30],
            [-0.20, 0.12, -0.30],
            [-0.20, -0.12, -0.30],
        ],
        dtype=np.float32,
    )
    foot_pos_w = foot_pos_b.copy()
    foot_pos_w[:, 2] += 0.32
    return ObsPacket(
        t=0.0,
        base_pos_w=np.array([0.0, 0.0, 0.32], dtype=np.float32),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        base_lin_vel_w=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        base_ang_vel_w=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(N_JOINTS, dtype=np.float32),
        joint_vel=np.zeros(N_JOINTS, dtype=np.float32),
        foot_pos_b=foot_pos_b,
        foot_pos_w=foot_pos_w,
        foot_contact=np.ones(N_FEET, dtype=bool),
        cmd_vel=np.array([0.2, 0.0, 0.0], dtype=np.float32),
        height_scan=None,
        last_action=np.zeros(N_JOINTS, dtype=np.float32),
        terrain_context=terrain_context,
    )


if __name__ == "__main__":
    main()

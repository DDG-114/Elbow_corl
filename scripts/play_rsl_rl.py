#!/usr/bin/env python3
"""Register project Isaac tasks, then launch Isaac Lab's RSL-RL player."""

from __future__ import annotations

from pathlib import Path
import runpy
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import go1_lewm_mpc.isaac_tasks  # noqa: F401

ISAACLAB_ROOT = Path("/home/kaga/IsaacLab")
RSL_RL_DIR = ISAACLAB_ROOT / "scripts" / "reinforcement_learning" / "rsl_rl"
PLAY_SCRIPT = RSL_RL_DIR / "play.py"

if str(RSL_RL_DIR) not in sys.path:
    sys.path.insert(0, str(RSL_RL_DIR))

runpy.run_path(str(PLAY_SCRIPT), run_name="__main__")

import numpy as np
import pytest

from go1_lewm_mpc.common.constants import N_JOINTS
from go1_lewm_mpc.controllers.gait_scheduler import GaitScheduler
from go1_lewm_mpc.controllers.go1_kinematics import (
    GO1_DEFAULT_JOINT_POS,
    GO1_JOINT_NAMES,
    Go1Kinematics,
    default_joint_pos_for_order,
    resolve_joint_order,
)
from go1_lewm_mpc.controllers.ik_position_controller import IKPositionController, q_to_isaac_action
from go1_lewm_mpc.controllers.swing_trajectory import SwingTrajectory
from go1_lewm_mpc.tests.fixtures import make_fake_mpc_plan, make_fake_obs_packet


def test_go1_ik_returns_finite_joint_positions_for_nominal_feet() -> None:
    obs = make_fake_obs_packet()
    q_des, clipped = Go1Kinematics().inverse_kinematics(obs.foot_pos_b)

    assert q_des.shape == (N_JOINTS,)
    assert clipped.shape == (4,)
    assert np.isfinite(q_des).all()
    assert q_des[2] < 0.0
    assert q_des[5] < 0.0


def test_q_to_isaac_action_uses_default_offset_and_scale() -> None:
    q_des = GO1_DEFAULT_JOINT_POS + 0.125
    action = q_to_isaac_action(q_des, q_default=GO1_DEFAULT_JOINT_POS, action_scale=0.25, max_action_abs=3.0)

    assert action.shape == (N_JOINTS,)
    assert np.allclose(action, 0.5)


def test_joint_order_mapping_reorders_defaults() -> None:
    runtime_names = list(reversed(GO1_JOINT_NAMES))
    order = resolve_joint_order(runtime_names)
    defaults = default_joint_pos_for_order(order)

    assert sorted(order.tolist()) == list(range(N_JOINTS))
    assert np.allclose(defaults, GO1_DEFAULT_JOINT_POS[::-1])


def test_gait_scheduler_cycles_crawl_order_without_contacts() -> None:
    scheduler = GaitScheduler(swing_duration=0.25, stance_duration=0.40)

    legs = [scheduler.update(t).swing_leg_id for t in (0.0, 0.66, 1.31, 1.96)]

    assert legs == [0, 3, 1, 2]


def test_swing_trajectory_starts_ends_and_clears_midpoint() -> None:
    traj = SwingTrajectory(flat_clearance=0.06, rough_clearance=0.10)
    start = np.array([0.2, 0.12, -0.30], dtype=np.float32)
    target = np.array([0.25, 0.14, -0.30], dtype=np.float32)

    first = traj.point(start, target, 0.0, terrain_phase="flat")
    mid = traj.point(start, target, 0.5, terrain_phase="rough")
    last = traj.point(start, target, 1.0, terrain_phase="flat")

    assert np.allclose(first, start)
    assert np.allclose(last, target)
    assert mid[2] > start[2]
    assert mid[2] == pytest.approx(-0.20)


def test_ik_position_controller_outputs_finite_clipped_12d_action() -> None:
    obs = make_fake_obs_packet()
    gait = GaitScheduler().update(obs.t)
    plan = make_fake_mpc_plan()
    controller = IKPositionController(max_action_abs=3.0, max_q_delta=0.08)

    body_plan = controller.make_body_plan(obs, gait, plan, terrain_phase="flat")
    packet = controller.compute_action(obs, body_plan, joint_names=GO1_JOINT_NAMES)

    assert body_plan.foot_targets_b.shape == (4, 3)
    assert packet.q_des.shape == (N_JOINTS,)
    assert packet.raw_action.shape == (N_JOINTS,)
    assert np.isfinite(packet.raw_action).all()
    assert np.max(np.abs(packet.raw_action)) <= 3.0
    assert not np.allclose(packet.raw_action, 0.0)

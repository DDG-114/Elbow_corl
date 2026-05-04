import numpy as np
import pytest

from go1_lewm_mpc.common.constants import FOOT_ORDER, N_FEET, N_JOINTS
from go1_lewm_mpc.common.types import (
    FootholdCandidatePacket,
    LatentPacket,
    LowLevelCue,
    MpcPlanPacket,
    ObsPacket,
)


def make_obs_packet() -> ObsPacket:
    return ObsPacket(
        t=0.02,
        base_pos_w=np.zeros(3),
        base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        base_lin_vel_w=np.zeros(3),
        base_ang_vel_w=np.zeros(3),
        joint_pos=np.zeros(N_JOINTS),
        joint_vel=np.zeros(N_JOINTS),
        foot_pos_b=np.zeros((N_FEET, 3)),
        foot_pos_w=np.zeros((N_FEET, 3)),
        foot_contact=np.array([True, False, True, False]),
        cmd_vel=np.array([0.2, 0.0, 0.1]),
        height_scan=None,
        last_action=np.zeros(N_JOINTS),
        payload_mass=1.0,
        payload_com_b=np.zeros(3),
    )


def test_constants_are_go1_phase_one_contract() -> None:
    assert FOOT_ORDER == ["FL", "FR", "RL", "RR"]
    assert N_FEET == 4
    assert N_JOINTS == 12


def test_obs_packet_accepts_valid_shapes_and_casts_arrays() -> None:
    obs = make_obs_packet()

    assert obs.base_pos_w.shape == (3,)
    assert obs.base_quat_wxyz.shape == (4,)
    assert obs.joint_pos.shape == (12,)
    assert obs.foot_pos_b.shape == (4, 3)
    assert obs.foot_contact.dtype == np.bool_
    assert obs.cmd_vel.dtype == np.float32


def test_obs_packet_rejects_invalid_joint_shape() -> None:
    with pytest.raises(ValueError, match="joint_pos"):
        ObsPacket(
            t=0.0,
            base_pos_w=np.zeros(3),
            base_quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
            base_lin_vel_w=np.zeros(3),
            base_ang_vel_w=np.zeros(3),
            joint_pos=np.zeros(11),
            joint_vel=np.zeros(N_JOINTS),
            foot_pos_b=np.zeros((N_FEET, 3)),
            foot_pos_w=np.zeros((N_FEET, 3)),
            foot_contact=np.zeros(N_FEET),
            cmd_vel=np.zeros(3),
            height_scan=None,
            last_action=None,
        )


def test_latent_packet_validates_finite_values() -> None:
    packet = LatentPacket(
        t=0.0,
        z=np.zeros(4),
        terrain_feat=np.ones(2),
        dyn_feat=np.ones(3),
        uncertainty=0.25,
    )

    assert packet.z.dtype == np.float32
    assert packet.uncertainty == pytest.approx(0.25)

    with pytest.raises(ValueError, match="z"):
        LatentPacket(
            t=0.0,
            z=np.array([np.nan]),
            terrain_feat=np.ones(2),
            dyn_feat=np.ones(3),
            uncertainty=0.25,
        )


def test_foothold_candidate_packet_validates_candidate_lengths() -> None:
    packet = FootholdCandidatePacket(
        t=0.1,
        swing_leg_id=0,
        candidates_b=np.zeros((3, 3)),
        candidates_w=np.ones((3, 3)),
        risk=np.array([0.1, 0.2, 0.3]),
        reach_cost=np.array([0.0, 0.1, 0.2]),
        total_score=np.array([0.1, 0.3, 0.5]),
    )

    assert packet.candidates_b.shape == (3, 3)
    assert packet.risk.shape == (3,)

    with pytest.raises(ValueError, match="risk"):
        FootholdCandidatePacket(
            t=0.1,
            swing_leg_id=0,
            candidates_b=np.zeros((3, 3)),
            candidates_w=np.ones((3, 3)),
            risk=np.array([0.1, 0.2]),
            reach_cost=np.array([0.0, 0.1, 0.2]),
            total_score=np.array([0.1, 0.3, 0.5]),
        )


def test_mpc_plan_packet_and_low_level_cue_shapes() -> None:
    plan = MpcPlanPacket(
        t=0.2,
        selected_leg_id=3,
        selected_foothold_b=np.zeros(3),
        selected_foothold_w=np.ones(3),
        velocity_bias=np.array([0.01, 0.0, -0.02]),
        confidence=0.8,
        debug={"solver": "stub"},
    )
    cue = LowLevelCue(
        cmd_vel_corrected=np.array([0.25, 0.0, 0.1]),
        foothold_hint_b=np.zeros((N_FEET, 3)),
        risk_summary=np.zeros(N_FEET),
    )

    assert plan.velocity_bias.shape == (3,)
    assert cue.foothold_hint_b.shape == (4, 3)


def test_leg_ids_must_be_in_foot_order_range() -> None:
    with pytest.raises(ValueError, match="selected_leg_id"):
        MpcPlanPacket(
            t=0.0,
            selected_leg_id=4,
            selected_foothold_b=np.zeros(3),
            selected_foothold_w=np.zeros(3),
            velocity_bias=np.zeros(3),
            confidence=1.0,
        )

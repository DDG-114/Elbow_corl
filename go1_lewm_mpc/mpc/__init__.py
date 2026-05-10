"""Foothold MPC and selector utilities."""

from go1_lewm_mpc.mpc.osqp_foothold import OSQPFootholdSelector
from go1_lewm_mpc.mpc.terrain_aware_selector import TerrainAwareFootholdSelector

__all__ = ["OSQPFootholdSelector", "TerrainAwareFootholdSelector"]

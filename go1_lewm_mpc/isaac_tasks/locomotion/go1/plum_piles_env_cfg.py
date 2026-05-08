"""Plum-pile / stepping-post terrain task for Unitree Go1 in Isaac Lab."""

from __future__ import annotations

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import (
    UnitreeGo1RoughEnvCfg,
)


GO1_PLUM_PILES_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=11,
    curriculum=False,
    size=(8.0, 8.0),
    border_width=2.0,
    num_rows=1,
    num_cols=1,
    horizontal_scale=0.10,
    vertical_scale=0.005,
    slope_threshold=0.75,
    color_scheme="height",
    use_cache=False,
    sub_terrains={
        "plum_piles": terrain_gen.MeshRepeatedCylindersTerrainCfg(
            proportion=1.0,
            platform_width=1.2,
            platform_height=0.22,
            abs_height_noise=(0.0, 0.0),
            rel_height_noise=(0.04 / 0.22, 1.0),
            object_params_start=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=220,
                height=0.22,
                radius=0.10,
                max_yx_angle=0.0,
                degrees=True,
            ),
            object_params_end=terrain_gen.MeshRepeatedCylindersTerrainCfg.ObjectCfg(
                num_objects=220,
                height=0.22,
                radius=0.10,
                max_yx_angle=0.0,
                degrees=True,
            ),
        ),
    },
)


@configclass
class UnitreeGo1PlumPilesEnvCfg(UnitreeGo1RoughEnvCfg):
    """Unitree Go1 velocity-tracking task on fixed plum-pile terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = GO1_PLUM_PILES_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = None
        self.actions.joint_pos.scale = 0.25

        self.commands.base_velocity.ranges.lin_vel_x = (-0.1, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)

        if hasattr(self, "events") and hasattr(self.events, "push_robot"):
            self.events.push_robot = None


@configclass
class UnitreeGo1PlumPilesEnvCfg_PLAY(UnitreeGo1PlumPilesEnvCfg):
    """Single-env play config for plum-pile visualization and rollout."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False

        if hasattr(self, "events"):
            if hasattr(self.events, "base_external_force_torque"):
                self.events.base_external_force_torque = None
            if hasattr(self.events, "push_robot"):
                self.events.push_robot = None

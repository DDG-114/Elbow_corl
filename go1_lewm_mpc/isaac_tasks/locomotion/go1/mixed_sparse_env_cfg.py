"""Mixed sparse terrain curriculum for Unitree Go1 in Isaac Lab."""

from __future__ import annotations

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import (
    UnitreeGo1RoughEnvCfg,
)


GO1_MIXED_SPARSE_TERRAINS_CFG = terrain_gen.TerrainGeneratorCfg(
    seed=42,
    curriculum=True,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.10,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(
            proportion=0.10,
        ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.15,
            noise_range=(0.01, 0.07),
            noise_step=0.01,
            border_width=0.25,
        ),
        "stepping_stones": terrain_gen.HfSteppingStonesTerrainCfg(
            proportion=0.20,
            stone_height_max=0.10,
            stone_width_range=(0.24, 0.36),
            stone_distance_range=(0.08, 0.24),
            holes_depth=-0.30,
            platform_width=1.2,
            border_width=0.25,
        ),
        "gaps": terrain_gen.MeshGapTerrainCfg(
            proportion=0.15,
            gap_width_range=(0.12, 0.32),
            platform_width=1.2,
        ),
        "pits": terrain_gen.MeshPitTerrainCfg(
            proportion=0.10,
            pit_depth_range=(0.04, 0.16),
            platform_width=1.2,
            double_pit=False,
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.15,
            grid_width=0.38,
            grid_height_range=(0.03, 0.12),
            platform_width=1.2,
            holes=False,
        ),
        "rails": terrain_gen.MeshRailsTerrainCfg(
            proportion=0.10,
            rail_thickness_range=(0.18, 0.28),
            rail_height_range=(0.04, 0.12),
            platform_width=1.2,
        ),
        "rings": terrain_gen.MeshFloatingRingTerrainCfg(
            proportion=0.05,
            ring_width_range=(0.45, 0.80),
            ring_height_range=(0.04, 0.12),
            ring_thickness=0.18,
            platform_width=1.2,
        ),
    },
)


@configclass
class UnitreeGo1MixedSparseEnvCfg(UnitreeGo1RoughEnvCfg):
    """Unitree Go1 velocity-tracking task with mixed sparse terrain."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = GO1_MIXED_SPARSE_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2
        self.actions.joint_pos.scale = 0.25

        if hasattr(self, "events") and hasattr(self.events, "push_robot"):
            self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (-0.2, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)


@configclass
class UnitreeGo1MixedSparseEnvCfg_PLAY(UnitreeGo1MixedSparseEnvCfg):
    """Smaller scene for visualization and policy rollout."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None

        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False

        if hasattr(self, "events"):
            if hasattr(self.events, "base_external_force_torque"):
                self.events.base_external_force_torque = None
            if hasattr(self.events, "push_robot"):
                self.events.push_robot = None

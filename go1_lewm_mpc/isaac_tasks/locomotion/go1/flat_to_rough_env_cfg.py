"""Fixed flat-to-rough terrain task for A2 Unitree Go1 IK rollouts."""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.utils import configclass
import isaaclab.terrains as terrain_gen

from isaaclab_tasks.manager_based.locomotion.velocity.config.go1.rough_env_cfg import (
    UnitreeGo1RoughEnvCfg,
)


@configclass
class UnitreeGo1FlatToRoughEnvCfg(UnitreeGo1RoughEnvCfg):
    """Unitree Go1 velocity task on one fixed flat-to-rough tile."""

    def __post_init__(self):
        super().__post_init__()

        terrain_cfg = _make_flat_to_rough_terrain_generator_cfg()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = terrain_cfg
        self.scene.terrain.max_init_terrain_level = None
        self.actions.joint_pos.scale = 0.25
        self.commands.base_velocity.ranges.lin_vel_x = (0.10, 0.15)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)

        if hasattr(self, "curriculum"):
            self.curriculum.terrain_levels = None
        if hasattr(self, "events"):
            if hasattr(self.events, "push_robot"):
                self.events.push_robot = None
            if hasattr(self.events, "base_external_force_torque"):
                self.events.base_external_force_torque = None


@configclass
class UnitreeGo1FlatToRoughEnvCfg_PLAY(UnitreeGo1FlatToRoughEnvCfg):
    """Single-env play config for A2 flat-to-rough rollouts."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False


def flat_to_rough_terrain(difficulty: float, cfg) -> tuple[list, object]:
    """Build one flat start region followed by a low-amplitude rough field."""

    import numpy as np
    import trimesh

    del difficulty
    size_x, size_y = float(cfg.size[0]), float(cfg.size[1])
    center_x, center_y = size_x * 0.5, size_y * 0.5
    transition_x = center_x - size_x * 0.5 + float(cfg.transition_x)
    cell = float(cfg.rough_cell)
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0) or 0))
    meshes = []

    flat_len = max(float(cfg.transition_x), 0.5)
    rough_len = max(size_x - flat_len, 0.5)
    meshes.append(_box_mesh(trimesh, flat_len, size_y, 0.02, (center_x - rough_len * 0.5, center_y)))

    x_values = np.arange(transition_x + cell * 0.5, center_x + size_x * 0.5, cell)
    y_values = np.arange(center_y - size_y * 0.5 + cell * 0.5, center_y + size_y * 0.5, cell)
    for x in x_values:
        for y in y_values:
            height = float(rng.uniform(float(cfg.rough_height_min), float(cfg.rough_height_max)))
            meshes.append(_box_mesh(trimesh, cell * 0.98, cell * 0.98, height, (float(x), float(y))))

    origin = np.asarray((center_x - size_x * 0.5 + 0.5, center_y, 0.05), dtype=np.float32)
    return meshes, origin


def _make_flat_to_rough_terrain_generator_cfg():
    cfg_cls = _make_flat_to_rough_cfg_class()
    return terrain_gen.TerrainGeneratorCfg(
        seed=19,
        curriculum=False,
        size=(8.0, 4.0),
        border_width=2.0,
        border_height=0.5,
        num_rows=1,
        num_cols=1,
        horizontal_scale=0.05,
        vertical_scale=0.005,
        slope_threshold=0.75,
        color_scheme="height",
        use_cache=False,
        sub_terrains={
            "flat_to_rough": cfg_cls(
                proportion=1.0,
                transition_x=2.0,
                rough_height_min=0.01,
                rough_height_max=0.05,
                rough_cell=0.20,
                seed=19,
            )
        },
    )


def _make_flat_to_rough_cfg_class():
    from isaaclab.terrains import SubTerrainBaseCfg

    @configclass
    class FlatToRoughTerrainCfg(SubTerrainBaseCfg):
        """One mesh tile with flat start and rough continuation."""

        function = flat_to_rough_terrain

        transition_x: float = MISSING
        rough_height_min: float = MISSING
        rough_height_max: float = MISSING
        rough_cell: float = MISSING
        seed: int = 19

    return FlatToRoughTerrainCfg


def _box_mesh(trimesh, length: float, width: float, top_z: float, center_xy: tuple[float, float]):
    height = max(float(top_z), 0.02)
    center = (float(center_xy[0]), float(center_xy[1]), height * 0.5)
    return trimesh.creation.box(
        (float(length), float(width), height),
        trimesh.transformations.translation_matrix(center),
    )

from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.usd_utils import add_ground_slab


class FlatPatch(Obstacle):
    """The bare tile (`tile.width` x `tile.depth`) with no obstacle at all —
    a rest/baseline tile. `diff.height` and `extra` are unused.
    """

    name = "flat_patch"

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        return self.flat_heightmap(tile)

from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

PLATFORM_WIDTH = 2.0  # default raised platform width in the middle
# platform is centered in the tile — flat margin on each side is whatever's
# left of tile.width (1.5m each by default: (5.0 - 2.0) / 2)


class RaisedPlatform(Obstacle):
    """Left to right: a flat spawn zone, a raised platform CENTERED in the
    tile, then a flat goal zone (both margins equal). `diff.height` is the
    platform's rise above the ground. `extra.platform_width` overrides the
    default above (2.0m).
    """

    name = "raised_platform"

    def _platform_width(self, diff: DifficultyParams) -> float:
        return diff.extra.get("platform_width", PLATFORM_WIDTH)

    def _platform_x(self, tile: TileSpec, diff: DifficultyParams) -> float:
        return 0.0

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        add_ground_slab(
            stage, f"{prim_path}/platform", self._platform_x(tile, diff), 0.0, self._platform_width(diff),
            tile.depth, tile.base_z + diff.height, tile,
        )

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        paint_rect(
            h, tile, self._platform_x(tile, diff), 0.0, self._platform_width(diff), tile.depth,
            tile.base_z + diff.height,
        )
        return h

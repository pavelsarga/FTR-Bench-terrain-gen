from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

FRONT_MARGIN = 1.5  # default flat spawn zone at the start of the tile
PLATFORM_WIDTH = 2.0  # default raised platform width in the middle
# back margin (flat goal zone) is whatever's left of tile.width — 1.5m by
# default (FRONT_MARGIN + PLATFORM_WIDTH + 1.5 == the default tile.width of 5.0)


class RaisedPlatform(Obstacle):
    """Left to right: a flat spawn zone, a raised platform spanning the
    full lane width, then a flat goal zone. `diff.height` is the platform's
    rise above the ground. `extra.front_margin`/`extra.platform_width`
    override the defaults above (1.5m / 2.0m).
    """

    name = "raised_platform"

    def _front_margin(self, diff: DifficultyParams) -> float:
        return diff.extra.get("front_margin", FRONT_MARGIN)

    def _platform_width(self, diff: DifficultyParams) -> float:
        return diff.extra.get("platform_width", PLATFORM_WIDTH)

    def _platform_x(self, tile: TileSpec, diff: DifficultyParams) -> float:
        return -tile.width / 2 + self._front_margin(diff) + self._platform_width(diff) / 2

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

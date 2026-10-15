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

    `extra.exit_height_ratio` r (default 1.0) splits the platform into two equal halves along the
    path, at `diff.height` (the half at lower X) and r * `diff.height`: the
    robot climbs h, steps by (r - 1) h, then drops r h, so the drop is no longer a replay of the
    rise (driven the other way: climbs r h, drops h). Usually drawn per tile via `extra.sample`.
    """

    name = "raised_platform"

    def _platform_width(self, diff: DifficultyParams) -> float:
        return diff.extra.get("platform_width", PLATFORM_WIDTH)

    def _platform_x(self, tile: TileSpec, diff: DifficultyParams) -> float:
        return 0.0

    def _tiers(self, tile: TileSpec, diff: DifficultyParams):
        """(x_center, width, top_z) of the platform slab(s): one, or two halves when r != 1."""
        x, w = self._platform_x(tile, diff), self._platform_width(diff)
        r = float(diff.extra.get("exit_height_ratio", 1.0))
        if abs(r - 1.0) < 1e-6:
            return [(x, w, tile.base_z + diff.height)]
        half = round(w / 2 / tile.cell_size) * tile.cell_size
        return [(x - w / 2 + half / 2, half, tile.base_z + diff.height),
                (x + half / 2, w - half, tile.base_z + r * diff.height)]

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        for k, (x, w, top) in enumerate(self._tiers(tile, diff)):
            name = "platform" if k == 0 else f"platform_{k}"
            add_ground_slab(stage, f"{prim_path}/{name}", x, 0.0, w, tile.depth, top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for x, w, top in self._tiers(tile, diff):
            paint_rect(h, tile, x, 0.0, w, tile.depth, top)
        return h

from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

LOG_LENGTH = 2.0  # default log length along the travel direction (X) — FIXED, not graded
LOG_WIDTH = 1.0  # default log width across the lane (Y) — FIXED, not graded
# the log is centered in the tile — flat margins on each side are equal


class LogCrossing(Obstacle):
    """A raised rectangular block lying with its long axis ALONG the
    direction of travel (X) — unlike every other obstacle here, which spans
    across the lane — so the robot mounts one end and rides along the top
    for its length, rather than climbing over a barrier. CENTERED in the
    tile (both margins equal). Only `diff.height` (the block's height above
    ground) grades across repeats; `log_length` and `log_width` are fixed
    geometry, not tied to difficulty. `extra.log_length`/`extra.log_width`
    override the defaults above (2.0m / 1.0m).
    """

    name = "log_crossing"

    def _log_x(self, tile: TileSpec, diff: DifficultyParams) -> float:
        return 0.0

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        log_length = diff.extra.get("log_length", LOG_LENGTH)
        log_width = diff.extra.get("log_width", LOG_WIDTH)
        add_ground_slab(
            stage, f"{prim_path}/log", self._log_x(tile, diff), 0.0, log_length, log_width,
            tile.base_z + diff.height, tile,
        )

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        log_length = diff.extra.get("log_length", LOG_LENGTH)
        log_width = diff.extra.get("log_width", LOG_WIDTH)
        paint_rect(h, tile, self._log_x(tile, diff), 0.0, log_length, log_width, tile.base_z + diff.height)
        return h

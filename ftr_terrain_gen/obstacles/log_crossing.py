from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

FRONT_MARGIN = 1.5  # default flat spawn zone at the start of the tile
LOG_LENGTH = 2.0  # default log length along the travel direction (X) — FIXED, not graded
LOG_WIDTH = 1.0  # default log width across the lane (Y) — FIXED, not graded
# back margin (flat goal zone) is whatever's left of tile.width — 1.5m by
# default (FRONT_MARGIN + LOG_LENGTH + 1.5 == the default tile.width of 5.0)


class LogCrossing(Obstacle):
    """A raised rectangular block lying with its long axis ALONG the
    direction of travel (X) — unlike every other obstacle here, which spans
    across the lane — so the robot mounts one end and rides along the top
    for its length, rather than climbing over a barrier. Only `diff.height`
    (the block's height above ground) grades across repeats; `log_length`
    and `log_width` are fixed geometry, not tied to difficulty.
    `extra.front_margin`/`extra.log_length`/`extra.log_width` override the
    defaults above (1.5m / 2.0m / 1.0m).
    """

    name = "log_crossing"

    def _log_x(self, tile: TileSpec, diff: DifficultyParams) -> float:
        front_margin = diff.extra.get("front_margin", FRONT_MARGIN)
        log_length = diff.extra.get("log_length", LOG_LENGTH)
        return -tile.width / 2 + front_margin + log_length / 2

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

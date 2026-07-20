from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect, symmetric_stair_profile
from ftr_terrain_gen.usd_utils import add_ground_slab

FRONT_MARGIN = 1.0  # default flat spawn zone at the start of the tile
STEP_LENGTH = 0.3  # default tread depth per step — FIXED, not graded
PLATFORM_LENGTH = 1.0  # default flat platform on top
N_STEPS = 5  # default number of steps on each side (up, then down)
# back margin (flat goal zone) is whatever's left of tile.width — 0.5m by
# default (FRONT_MARGIN + 2*N_STEPS*STEP_LENGTH + PLATFORM_LENGTH + 0.5
# == the default tile.width of 5.0)


class RaisedStairs(Obstacle):
    """Left to right: a flat spawn zone, `n_steps` steps up, a flat
    platform, `n_steps` steps back down (mirrored), then a flat goal zone.
    Step tread depth (`step_length`) is FIXED — only the RISE per step
    (and therefore the platform's total height, `diff.height`) grades
    across repeats. `extra.front_margin`/`extra.step_length`/
    `extra.platform_length`/`extra.n_steps` override the defaults above.
    """

    name = "raised_stairs"

    def _profile(self, tile: TileSpec, diff: DifficultyParams):
        return symmetric_stair_profile(
            tile,
            front_margin=diff.extra.get("front_margin", FRONT_MARGIN),
            step_length=diff.extra.get("step_length", STEP_LENGTH),
            platform_length=diff.extra.get("platform_length", PLATFORM_LENGTH),
            n_steps=int(diff.extra.get("n_steps", N_STEPS)),
            height=diff.height,
            sign=1.0,
        )

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        for i, (x, w, top) in enumerate(self._profile(tile, diff)):
            add_ground_slab(stage, f"{prim_path}/seg_{i}", x, 0.0, w, tile.depth, top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for x, w, top in self._profile(tile, diff):
            paint_rect(h, tile, x, 0.0, w, tile.depth, top)
        return h

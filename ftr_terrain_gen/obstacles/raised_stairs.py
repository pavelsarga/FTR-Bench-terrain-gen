from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect, symmetric_stair_profile
from ftr_terrain_gen.usd_utils import add_ground_slab

STEP_LENGTH = 0.3  # default tread depth per step — FIXED, not graded
PLATFORM_LENGTH = 1.0  # default flat platform on top
N_STEPS = 5  # default number of steps on each side (up, then down)
# the whole staircase is centered in the tile — flat margins on each side
# are equal, whatever's left of tile.width


class RaisedStairs(Obstacle):
    """Left to right: a flat spawn zone, `n_steps` steps up, a flat
    platform, `n_steps` steps back down (mirrored), then a flat goal zone —
    CENTERED in the tile (both margins equal). Step tread depth
    (`step_length`) is FIXED — only the RISE per step (and therefore the
    platform's total height, `diff.height`) grades across repeats.
    `extra.step_length`/`extra.platform_length`/`extra.n_steps` override
    the defaults above.
    """

    name = "raised_stairs"

    def _profile(self, tile: TileSpec, diff: DifficultyParams):
        step_length = diff.extra.get("step_length", STEP_LENGTH)
        platform_length = diff.extra.get("platform_length", PLATFORM_LENGTH)
        n_steps = int(diff.extra.get("n_steps", N_STEPS))
        margin = (tile.width - 2 * n_steps * step_length - platform_length) / 2
        return symmetric_stair_profile(
            tile,
            front_margin=margin,
            step_length=step_length,
            platform_length=platform_length,
            n_steps=n_steps,
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

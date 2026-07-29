from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect, value_noise_2d
from ftr_terrain_gen.usd_utils import add_ground_slab

FIELD_SIZE = 3.0  # default formation field extent (X and Y)
GRID_N = 7  # default grid resolution — coarser than cobblestones for a chunky, low-poly look
N_LEVELS = 4  # default number of discrete height tiers (not continuous — gives stepped terracing)
COARSE_N = 3  # default resolution of the underlying random control-point grid before smoothing
DROPOUT = 0.15  # default noise threshold below which a cell is omitted entirely (stays flat) —
# this is what breaks up the field's outline into an irregular, ragged footprint
# the field is centered in the tile — flat margins on each side are equal


class RockFormation(Obstacle):
    """Like `cobblestones`, but built from SPATIALLY CORRELATED noise
    (`shapes.value_noise_2d`, bilinearly-upsampled from a coarse random
    control grid) instead of independent per-cell randomness, quantized
    into `n_levels` discrete height tiers — so neighboring cells tend to
    share similar heights, giving a cohesive stepped/terraced look instead
    of checkerboard static. Cells whose noise value falls below `dropout`
    are omitted entirely (left flat), which breaks the field's outline into
    an irregular, ragged footprint rather than a filled rectangle. CENTERED
    in the tile (both margins equal).
    `extra.field_size`/`extra.grid_n`/`extra.n_levels`/`extra.coarse_n`/
    `extra.dropout` override the defaults above.
    """

    name = "rock_formation"

    def _field(self, diff: DifficultyParams):
        field_size = diff.extra.get("field_size", FIELD_SIZE)
        grid_n = int(diff.extra.get("grid_n", GRID_N))
        n_levels = int(diff.extra.get("n_levels", N_LEVELS))
        coarse_n = int(diff.extra.get("coarse_n", COARSE_N))
        dropout = diff.extra.get("dropout", DROPOUT)

        cell = field_size / grid_n
        noise = value_noise_2d(diff.rng(), grid_n, grid_n, coarse_n)
        levels = np.floor(noise * n_levels).clip(0, n_levels - 1)
        heights = levels / (n_levels - 1) * diff.height
        active = noise >= dropout
        return field_size, grid_n, cell, heights, active

    def _field_x(self, tile: TileSpec, diff: DifficultyParams, field_size: float) -> float:
        return 0.0

    def _cells(self, tile: TileSpec, diff: DifficultyParams):
        field_size, grid_n, cell, heights, active = self._field(diff)
        x0 = self._field_x(tile, diff, field_size) - field_size / 2
        y0 = -field_size / 2
        for i in range(grid_n):
            for j in range(grid_n):
                if not active[i, j]:
                    continue
                cx = x0 + cell * (i + 0.5)
                cy = y0 + cell * (j + 0.5)
                yield cx, cy, cell, tile.base_z + heights[i, j]

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        for k, (cx, cy, cell, top) in enumerate(self._cells(tile, diff)):
            add_ground_slab(stage, f"{prim_path}/cell_{k}", cx, cy, cell, cell, top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for cx, cy, cell, top in self._cells(tile, diff):
            paint_rect(h, tile, cx, cy, cell, cell, top)
        return h

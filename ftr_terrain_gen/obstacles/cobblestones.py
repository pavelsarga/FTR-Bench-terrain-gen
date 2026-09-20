from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

FIELD_SIZE = 3.0  # default cobblestone field extent (X and Y) — a "3x3m part of the base"
GRID_N = 11  # default grid resolution (grid_n x grid_n cells)
# the field is centered in the tile — flat margins on each side are equal


class Cobblestones(Obstacle):
    """A `field_size` x `field_size` (default 3m x 3m) patch, CENTERED in
    the tile, split into a `grid_n` x `grid_n` (default 11x11) grid of
    small raised blocks. Each cell's height is independently randomized
    (seeded, reproducible from `diff.seed`) between 0 and `diff.height` —
    so the field gets rougher, not just uniformly taller, as difficulty
    increases. `extra.field_size`/`extra.grid_n` override the defaults
    above.
    """

    name = "cobblestones"

    def _grid(self, diff: DifficultyParams):
        field_size = diff.extra.get("field_size", FIELD_SIZE)
        grid_n = int(diff.extra.get("grid_n", GRID_N))
        cell = field_size / grid_n
        heights = diff.rng().uniform(0.0, diff.height, size=(grid_n, grid_n))
        max_step = diff.extra.get("max_step")
        if max_step is not None:
            # NIST stepfield rule: no two adjacent blocks differ by more than
            # `max_step` — raise the low neighbour of any too-tall block.
            # Without it a field's difficulty is decided by the seed, not by
            # `diff.height` (custom_mixed's cobblestones scored 0.39 at col 7
            # and 0.71 at col 8), and a single 0.3 m block next to a 0 m one
            # is a step the robot cannot take from inside the field.
            for _ in range(grid_n):
                changed = False
                for i in range(grid_n):
                    for j in range(grid_n):
                        for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                            if 0 <= a < grid_n and 0 <= b < grid_n and heights[a, b] - heights[i, j] > max_step:
                                heights[i, j] = heights[a, b] - max_step
                                changed = True
                if not changed:
                    break
        return field_size, grid_n, cell, heights

    def _field_x(self, tile: TileSpec, diff: DifficultyParams, field_size: float) -> float:
        return 0.0

    def _cells(self, tile: TileSpec, diff: DifficultyParams):
        field_size, grid_n, cell, heights = self._grid(diff)
        x0 = self._field_x(tile, diff, field_size) - field_size / 2
        y0 = -field_size / 2
        for i in range(grid_n):
            for j in range(grid_n):
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

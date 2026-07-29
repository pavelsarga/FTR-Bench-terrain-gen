from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

FIELD_SIZE = 3.0  # default pyramid field extent (X and Y)
N_ROWS = 7  # default number of diagonal rows (MUST be odd — a single symmetric peak row)
HIDDEN_LAYERS = 0  # default number of outermost rows on EACH end left at height 0 (flush with
# the ground, no visible step) instead of getting a raised cube — e.g. hidden_layers=4 makes
# the 4 rows closest to each corner flat, giving a flat corner before the steps begin
# the field is centered in the tile — flat margins on each side are equal


class DiagonalPyramid(Obstacle):
    """A `field_size` x `field_size` (default 3m x 3m) square grid of
    cubes, where height is banded along the DIAGONAL (`i + j`): `n_rows`
    (default 7) diagonal rows rise from the two opposite corners to a
    single peak row through the center, then fall again — a diagonal
    pyramid. The diagonal's tilt (which pair of corners it rises from)
    mirrors across Y by repeat (`diff.col_index` parity), so it alternates
    left-to-right / right-to-left rather than always tilting the same way.
    `hidden_layers` (default 0) outermost rows on EACH end are
    left at height 0 (flush with the ground, no cube at all) instead of
    getting a step — e.g. `hidden_layers: 4` leaves the 4 rows nearest each
    corner completely flat, so the pyramid only starts rising partway in.
    Among the rows that ARE stepped, the first one still sits one rise
    above the ground (not flush with it), so every stepped row reads as a
    visibly distinct step rather than blending into the flat corner.
    Row spacing/count is fixed; only the RISE per (visible) row grades
    with difficulty, same convention as raised_stairs/lowered_stairs.
    CENTERED in the tile (both margins equal).
    `extra.field_size`/`extra.n_rows`/`extra.hidden_layers` override the
    defaults above (`n_rows` must be odd; `hidden_layers` must leave at
    least the peak row visible).
    """

    name = "diagonal_pyramid"

    def _grid(self, diff: DifficultyParams):
        field_size = diff.extra.get("field_size", FIELD_SIZE)
        n_rows = int(diff.extra.get("n_rows", N_ROWS))
        hidden_layers = int(diff.extra.get("hidden_layers", HIDDEN_LAYERS))
        if n_rows % 2 == 0:
            raise ValueError(f"diagonal_pyramid's n_rows ({n_rows}) must be odd (a single symmetric peak row)")
        grid_n = (n_rows + 1) // 2  # e.g. n_rows=7 -> grid_n=4 (4x4 grid, diagonal bands 0..6)
        peak = grid_n - 1  # diagonal-band index of the peak row
        if hidden_layers < 0 or hidden_layers > peak:
            raise ValueError(f"diagonal_pyramid's hidden_layers ({hidden_layers}) must be in [0, {peak}]")
        # m = how many distinct step heights exist among the VISIBLE rows,
        # minus 1 (0 at the first visible row, m at the peak). +1 in the
        # rise formula so even the first visible row sits one rise above
        # the ground, not flush with it — a row at height 0 would be
        # visually indistinguishable from the flat corner/surroundings.
        m = peak - hidden_layers
        rise_per_row = diff.height / (m + 1)
        cell = field_size / grid_n
        return field_size, grid_n, cell, peak, hidden_layers, m, rise_per_row

    def _field_x(self, tile: TileSpec, diff: DifficultyParams, field_size: float) -> float:
        return 0.0

    def _cells(self, tile: TileSpec, diff: DifficultyParams):
        field_size, grid_n, cell, peak, hidden_layers, m, rise_per_row = self._grid(diff)
        x0 = self._field_x(tile, diff, field_size) - field_size / 2
        y0 = -field_size / 2
        mirror = diff.col_index % 2 == 1  # alternates the diagonal's tilt by repeat
        for i in range(grid_n):
            for j in range(grid_n):
                j_band = (grid_n - 1 - j) if mirror else j
                u = i + j_band
                if u < hidden_layers or u > 2 * peak - hidden_layers:
                    continue  # in the hidden band — flush with the ground, no cube
                distance = abs(u - peak)
                height = rise_per_row * (m - distance + 1)
                cx = x0 + cell * (i + 0.5)
                cy = y0 + cell * (j + 0.5)
                yield cx, cy, cell, tile.base_z + height

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        for k, (cx, cy, cell, top) in enumerate(self._cells(tile, diff)):
            add_ground_slab(stage, f"{prim_path}/cell_{k}", cx, cy, cell, cell, top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for cx, cy, cell, top in self._cells(tile, diff):
            paint_rect(h, tile, cx, cy, cell, cell, top)
        return h

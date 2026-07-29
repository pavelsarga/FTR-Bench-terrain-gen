from __future__ import annotations

import math

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

FIELD_SIZE = 3.0  # grid extent covering the bump, X and Y
GRID_N = 84  # grid resolution (grid_n x grid_n cells)
SCALE = 0.5  # meters per unit of the function's (x, y) argument — controls bump width
HEIGHT = 0.4  # fixed peak height — does not grade with difficulty
OFFSET_RANGE = 0.75  # max |Y offset| of the bump's center from the tile's centerline
MIN_CELL_HEIGHT = 0.005  # cells below this height are left flat (avoids z-fighting with the base)
STEEPNES = 4

_PEAK = 0.25  # f(0), used to normalize the peak to exactly HEIGHT


def _bump(r2: float) -> float:
    """f(r2) = sigmoid'(r2) = e^-r2 / (1 + e^-r2)^2 — a smooth radial bump peaking at r2=0."""
    e = math.exp(-r2)
    return e / (1.0 + e) ** 2


class Stump(Obstacle):
    """A smooth radial bump, f(x, y) = sigmoid'(x^2 + y^2) in scaled
    coordinates, approximated by a `grid_n` x `grid_n` grid of small raised
    blocks over a `field_size` x `field_size` patch. Height is fixed
    (`HEIGHT`); what grades with difficulty is the bump's center Y offset,
    sliding from slightly left of the tile's centerline (`diff.t` = 0) to
    slightly right (`diff.t` = 1). Field is CENTERED in the tile (both
    margins equal). `extra.field_size`/`extra.grid_n`/`extra.scale`/
    `extra.height`/`extra.offset_range` override the defaults above.
    """

    name = "stump"

    def _field_x(self, tile: TileSpec, diff: DifficultyParams, field_size: float) -> float:
        return 0.0

    def _center_y(self, diff: DifficultyParams) -> float:
        offset_range = diff.extra.get("offset_range", OFFSET_RANGE)
        return -offset_range + 2.0 * offset_range * diff.t

    def _cells(self, tile: TileSpec, diff: DifficultyParams):
        field_size = diff.extra.get("field_size", FIELD_SIZE)
        grid_n = int(diff.extra.get("grid_n", GRID_N))
        scale = diff.extra.get("scale", SCALE)
        height = diff.extra.get("height", HEIGHT)
        field_x = self._field_x(tile, diff, field_size)
        center_y = self._center_y(diff)
        cell = field_size / grid_n
        x0 = field_x - field_size / 2
        y0 = -field_size / 2
        for i in range(grid_n):
            for j in range(grid_n):
                cx = x0 + cell * (i + 0.5)
                cy = y0 + cell * (j + 0.5)
                dx = (cx - field_x) / scale
                dy = (cy - center_y) / scale
                r2 = dx * dx + dy * dy
                h = height * _bump(STEEPNES*r2) / _PEAK
                if h < MIN_CELL_HEIGHT:
                    continue
                yield cx, cy, cell, tile.base_z + h

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        for k, (cx, cy, cell, top) in enumerate(self._cells(tile, diff)):
            add_ground_slab(stage, f"{prim_path}/cell_{k}", cx, cy, cell, cell, top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for cx, cy, cell, top in self._cells(tile, diff):
            paint_rect(h, tile, cx, cy, cell, cell, top)
        return h

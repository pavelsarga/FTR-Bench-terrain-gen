from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

FRONT_MARGIN = 1.0  # flat spawn zone at the start of the tile
FIELD_SIZE = 3.0  # raised-patch extent, X and Y


class HalfPlatform(Obstacle):
    """Only half (split along Y, the lane width) of a `field_size` x
    `field_size` patch is raised — one track crosses flat ground while the
    other is elevated. Which half is raised alternates by repeat
    (`diff.col_index` parity), not randomly. `diff.height` (graded
    `min_height` -> `max_height`) is the raised half's height.
    `extra.front_margin`/`extra.field_size` override the defaults above.
    """

    name = "half_platform"

    def _field_x(self, tile: TileSpec, diff: DifficultyParams, field_size: float) -> float:
        front_margin = diff.extra.get("front_margin", FRONT_MARGIN)
        return -tile.width / 2 + front_margin + field_size / 2

    def _half(self, tile: TileSpec, diff: DifficultyParams):
        field_size = diff.extra.get("field_size", FIELD_SIZE)
        field_x = self._field_x(tile, diff, field_size)
        half_depth = field_size / 2
        sign = 1.0 if diff.col_index % 2 == 0 else -1.0
        half_y = sign * half_depth / 2
        return field_x, field_size, half_depth, half_y

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        field_x, field_size, half_depth, half_y = self._half(tile, diff)
        add_ground_slab(
            stage, f"{prim_path}/raised_half", field_x, half_y, field_size, half_depth,
            tile.base_z + diff.height, tile,
        )

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        field_x, field_size, half_depth, half_y = self._half(tile, diff)
        paint_rect(h, tile, field_x, half_y, field_size, half_depth, tile.base_z + diff.height)
        return h

from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect, x_segment_centers
from ftr_terrain_gen.usd_utils import add_ground_slab

FRONT_MARGIN = 1.0  # default flat spawn zone at the start of the tile
RAIL_WIDTH = 0.2  # default thickness of each rail along X (both rails equal)
GAP_WIDTH = 1.0  # default flat space between the two rails
# back margin (flat goal zone) is whatever's left of tile.width — 2.5m by
# default (FRONT_MARGIN + RAIL_WIDTH + GAP_WIDTH + RAIL_WIDTH + 2.5
# == the default tile.width of 5.0)
WALL_WIDTH = 0.1  # default thin guard-wall thickness on each side
WALL_HEIGHT = 0.15  # default guard-wall height ABOVE the rails' own top surface


class TwinRails(Obstacle):
    """Left to right: a flat spawn zone, a thin raised rail spanning the
    full lane width (at right angle to the direction of travel), a flat gap,
    a second rail (same thickness as the first), then a flat goal zone.
    `diff.height` grades both rails' height together. Two thin flat guard
    walls run along the Y edges of the tile, spanning the rails+gap section
    (from the first rail's leading edge to the second rail's trailing
    edge), taller than the rails, enclosing the crossing on both sides.
    `extra.front_margin`/`extra.rail_width`/`extra.gap_width`/
    `extra.wall_width`/`extra.wall_height` override the defaults above.
    """

    name = "twin_rails"

    def _rail_centers(self, tile: TileSpec, diff: DifficultyParams) -> tuple[float, float]:
        front_margin = diff.extra.get("front_margin", FRONT_MARGIN)
        rail_width = diff.extra.get("rail_width", RAIL_WIDTH)
        gap_width = diff.extra.get("gap_width", GAP_WIDTH)
        back_margin = tile.width - front_margin - 2 * rail_width - gap_width
        widths = [front_margin, rail_width, gap_width, rail_width, back_margin]
        centers = x_segment_centers(tile, widths)
        return centers[1], centers[3]

    def _wall_span(self, tile: TileSpec, diff: DifficultyParams) -> tuple[float, float]:
        rail_width = diff.extra.get("rail_width", RAIL_WIDTH)
        rail1_x, rail2_x = self._rail_centers(tile, diff)
        lo, hi = rail1_x - rail_width / 2, rail2_x + rail_width / 2
        return (lo + hi) / 2, hi - lo  # center, length

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        rail_width = diff.extra.get("rail_width", RAIL_WIDTH)
        top = tile.base_z + diff.height
        for i, x in enumerate(self._rail_centers(tile, diff)):
            add_ground_slab(stage, f"{prim_path}/rail_{i}", x, 0.0, rail_width, tile.depth, top, tile)

        wall_x, wall_length = self._wall_span(tile, diff)
        wall_width = diff.extra.get("wall_width", WALL_WIDTH)
        wall_top = top + diff.extra.get("wall_height", WALL_HEIGHT)
        wall_y = tile.depth / 2 - wall_width / 2
        add_ground_slab(stage, f"{prim_path}/wall_pos", wall_x, wall_y, wall_length, wall_width, wall_top, tile)
        add_ground_slab(stage, f"{prim_path}/wall_neg", wall_x, -wall_y, wall_length, wall_width, wall_top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        rail_width = diff.extra.get("rail_width", RAIL_WIDTH)
        top = tile.base_z + diff.height
        for x in self._rail_centers(tile, diff):
            paint_rect(h, tile, x, 0.0, rail_width, tile.depth, top)
        return h

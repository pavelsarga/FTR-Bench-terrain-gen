from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import MIN_EDGE_MARGIN, paint_rect, x_segment_centers
from ftr_terrain_gen.usd_utils import add_ground_slab

FRONT_MARGIN = 1.0  # default flat spawn zone at the start of the tile
TRENCH_WIDTH = 1.0  # default width of both trenches
MIDDLE_WIDTH = 1.0  # default level ground section between the two trenches
# back margin (flat goal zone) is whatever's left of tile.width — 1.0m by
# default (FRONT_MARGIN + TRENCH_WIDTH + MIDDLE_WIDTH + TRENCH_WIDTH + 1.0
# == the default tile.width of 5.0)


class DoubleTrench(Obstacle):
    """Left to right: a flat spawn zone, a trench, a level ground section,
    a second trench (same width as the first), then a flat goal zone.
    `diff.height` is the depth of BOTH trenches below the ground.
    `extra.front_margin`/`extra.trench_width`/`extra.middle_width` override
    the defaults above (1.0m each).
    """

    name = "double_trench"

    def _segments(self, tile: TileSpec, diff: DifficultyParams):
        front_margin = diff.extra.get("front_margin", FRONT_MARGIN)
        middle_width = diff.extra.get("middle_width", MIDDLE_WIDTH)
        trench_width = diff.extra.get("trench_width", TRENCH_WIDTH)
        back_margin = tile.width - front_margin - 2 * trench_width - middle_width
        widths = [front_margin, trench_width, middle_width, trench_width, back_margin]
        is_trench = [False, True, False, True, False]
        centers = x_segment_centers(tile, widths)
        return list(zip(centers, widths, is_trench))

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        beam_y = tile.depth / 2 - MIN_EDGE_MARGIN / 2
        for i, (x, w, is_trench) in enumerate(self._segments(tile, diff)):
            if is_trench:
                add_ground_slab(
                    stage, f"{prim_path}/seg_{i}", x, 0.0, w, tile.depth - 2 * MIN_EDGE_MARGIN,
                    tile.base_z - diff.height, tile,
                )
                add_ground_slab(stage, f"{prim_path}/seg_{i}_beam_pos", x, beam_y, w, MIN_EDGE_MARGIN, tile.base_z, tile)
                add_ground_slab(stage, f"{prim_path}/seg_{i}_beam_neg", x, -beam_y, w, MIN_EDGE_MARGIN, tile.base_z, tile)
            else:
                add_ground_slab(stage, f"{prim_path}/seg_{i}", x, 0.0, w, tile.depth, tile.base_z, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for x, w, is_trench in self._segments(tile, diff):
            if is_trench:
                # side beams stay at flat_heightmap's default base_z — nothing to paint there
                paint_rect(h, tile, x, 0.0, w, tile.depth - 2 * MIN_EDGE_MARGIN, tile.base_z - diff.height)
            else:
                paint_rect(h, tile, x, 0.0, w, tile.depth, tile.base_z)
        return h

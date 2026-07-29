from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import MIN_EDGE_MARGIN, paint_rect, x_segment_centers
from ftr_terrain_gen.usd_utils import add_ground_slab

TRENCH_DEPTH = 0.3  # default fixed depth — this obstacle grades WIDTH, not depth
MIN_WIDTH = 0.3  # default trench width at the easiest repeat (diff.t == 0)
MAX_WIDTH = 1.5  # default trench width at the hardest repeat (diff.t == 1)
# the trench is centered in the tile — margins on each side are equal and
# shrink together as the trench widens across repeats


class WideningTrench(Obstacle):
    """A single trench, CENTERED in the tile (both margins equal), whose
    WIDTH (not depth) is graded across repeats: `min_width` (default 0.3m)
    at the easiest repeat to `max_width` (default 1.5m) at the hardest.
    Depth is fixed at `trench_depth` (default 0.3m). Overridable via
    `extra.trench_depth`/`extra.min_width`/`extra.max_width`.

    Built as 3 non-overlapping segments along X (spawn/trench/goal), NOT a
    full-tile base slab plus an embedded trench cube — a full-width slab
    would always stay the tallest surface over its own footprint, making an
    embedded trench underneath it physically inert.
    """

    name = "widening_trench"

    def _width(self, diff: DifficultyParams) -> float:
        lo = diff.extra.get("min_width", MIN_WIDTH)
        hi = diff.extra.get("max_width", MAX_WIDTH)
        return lo + (hi - lo) * diff.t

    def _segments(self, tile: TileSpec, diff: DifficultyParams):
        width = self._width(diff)
        margin = (tile.width - width) / 2
        widths = [margin, width, margin]
        is_trench = [False, True, False]
        centers = x_segment_centers(tile, widths)
        return list(zip(centers, widths, is_trench))

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        depth = diff.extra.get("trench_depth", TRENCH_DEPTH)
        beam_y = tile.depth / 2 - MIN_EDGE_MARGIN / 2
        for i, (x, w, is_trench) in enumerate(self._segments(tile, diff)):
            if is_trench:
                add_ground_slab(
                    stage, f"{prim_path}/seg_{i}", x, 0.0, w, tile.depth - 2 * MIN_EDGE_MARGIN,
                    tile.base_z - depth, tile,
                )
                add_ground_slab(stage, f"{prim_path}/seg_{i}_beam_pos", x, beam_y, w, MIN_EDGE_MARGIN, tile.base_z, tile)
                add_ground_slab(stage, f"{prim_path}/seg_{i}_beam_neg", x, -beam_y, w, MIN_EDGE_MARGIN, tile.base_z, tile)
            else:
                add_ground_slab(stage, f"{prim_path}/seg_{i}", x, 0.0, w, tile.depth, tile.base_z, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        depth = diff.extra.get("trench_depth", TRENCH_DEPTH)
        h = self.flat_heightmap(tile)
        for x, w, is_trench in self._segments(tile, diff):
            if is_trench:
                paint_rect(h, tile, x, 0.0, w, tile.depth - 2 * MIN_EDGE_MARGIN, tile.base_z - depth)
            else:
                paint_rect(h, tile, x, 0.0, w, tile.depth, tile.base_z)
        return h

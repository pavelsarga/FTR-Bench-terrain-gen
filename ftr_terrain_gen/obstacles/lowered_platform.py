from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import MIN_EDGE_MARGIN, paint_rect, x_segment_centers
from ftr_terrain_gen.usd_utils import add_ground_slab

PIT_WIDTH = 1.6  # pit floor length along X — longer than the robot (1.28 m tip to tip),
# so it is entirely inside before it can start climbing out


class LoweredPlatform(Obstacle):
    """The recessed mirror of `raised_platform` (cur_mixed's
    "lowered_platform"): a flat pit `pit_width` long and `diff.height` deep,
    spanning the lane. Dropping in is the easy half; the robot then has to
    re-hook its front flippers on the far wall with the rear pair still on
    the pit floor — a sequenced posture change, and the step class where
    MARV's limit is lowest (cur_mixed: 0.35 m ok, 0.40 m never). Built from
    non-overlapping X segments like `widening_trench`, with the same
    MIN_EDGE_MARGIN side beams.
    """

    name = "lowered_platform"

    def _segments(self, tile: TileSpec, diff: DifficultyParams):
        width = diff.extra.get("pit_width", PIT_WIDTH)
        margin = (tile.width - width) / 2
        widths = [margin, width, margin]
        centers = x_segment_centers(tile, widths)
        return list(zip(centers, widths, [False, True, False]))

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        beam_y = tile.depth / 2 - MIN_EDGE_MARGIN / 2
        for i, (x, w, is_pit) in enumerate(self._segments(tile, diff)):
            if is_pit:
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
        for x, w, is_pit in self._segments(tile, diff):
            if is_pit:
                paint_rect(h, tile, x, 0.0, w, tile.depth - 2 * MIN_EDGE_MARGIN, tile.base_z - diff.height)
        return h

from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

OPENING = 0.9  # gap width in the wall (MARV is 0.57 m wide)
WALL_HEIGHT = 1.0  # well above the robot (0.24 body + flippers up) — a wall, not a step to try
WALL_THICKNESS = 0.2
MIN_OFFSET = 0.3  # lateral offset of the opening centre from the lane centreline, easiest repeat
MAX_OFFSET = 0.5  # ... hardest. Keep <= 0.5: the 1.05 m wide heightmap only sees |y| <= 0.525
N_GATES = 1  # 1 = offset gate; 2 = chicane (second gate with the opposite offset)
GATE_SPACING = 2.5  # X distance between the two gates of a chicane


class OffsetGate(Obstacle):
    """A wall across the lane with an opening that is NOT on the robot's
    line: to get through, the robot must turn toward the opening and then
    straighten again — the only obstacle in the set that needs yaw. With
    `n_gates: 2` a second wall follows with the opening on the other side
    (a chicane). Offset grades `min_offset` -> `max_offset` with `diff.t`
    and flips sign on odd repeats when `extra.mirror_alternate` is set.
    Pair with the row's `goal_lateral_offset` (birth.py) so the target sits
    behind the LAST opening; the walls themselves are plain boxes and there
    is no other terrain in the tile, so the turn happens on flat ground
    where MARV can actually pivot (with its flippers raised). Only used by
    the `*_full` course; the straight-drive course leaves this row out.
    """

    name = "offset_gate"

    def gates(self, diff: DifficultyParams):
        """Yield (x, opening_center_y) per gate — sign of the LAST gate's
        offset is what birth.py's goal offset should follow."""
        n = int(diff.extra.get("n_gates", N_GATES))
        lo, hi = diff.extra.get("min_offset", MIN_OFFSET), diff.extra.get("max_offset", MAX_OFFSET)
        offset = diff.mirror_sign * (lo + (hi - lo) * diff.t)
        spacing = diff.extra.get("gate_spacing", GATE_SPACING)
        xs = [0.0] if n == 1 else [spacing / 2 * (1 - 2 * k / (n - 1)) for k in range(n)]
        # first gate is the one the robot (driving -X, from +X) meets first: largest x
        return [(x, offset * (1 if k % 2 == 0 else -1)) for k, x in enumerate(xs)]

    def final_goal_offset(self, diff: DifficultyParams) -> float:
        return self.gates(diff)[-1][1]

    def _walls(self, tile: TileSpec, diff: DifficultyParams):
        opening = diff.extra.get("opening", OPENING)
        thickness = diff.extra.get("wall_thickness", WALL_THICKNESS)
        for x, cy in self.gates(diff):
            left_edge = cy + opening / 2
            right_edge = cy - opening / 2
            # wall from the opening's edge to the lane edge, each side
            top_len = tile.depth / 2 - left_edge
            bot_len = right_edge + tile.depth / 2
            if top_len > 0:
                yield x, left_edge + top_len / 2, thickness, top_len
            if bot_len > 0:
                yield x, right_edge - bot_len / 2, thickness, bot_len

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        top = tile.base_z + diff.extra.get("wall_height", WALL_HEIGHT)
        for k, (x, y, w, d) in enumerate(self._walls(tile, diff)):
            add_ground_slab(stage, f"{prim_path}/wall_{k}", x, y, w, d, top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        top = tile.base_z + diff.extra.get("wall_height", WALL_HEIGHT)
        for x, y, w, d in self._walls(tile, diff):
            paint_rect(h, tile, x, y, w, d, top)
        return h

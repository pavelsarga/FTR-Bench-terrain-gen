from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect, x_segment_centers
from ftr_terrain_gen.usd_utils import add_ground_slab

ENTRY_MARGIN = 2.2  # flat margin at the ground-level end of the ramp
STEP_LENGTH = 0.3  # tread depth per step — fixed, not graded
N_STEPS = 5  # number of steps from bottom to top
# the platform fills whatever remains of tile.width after the entry margin
# and steps, so it reaches all the way to the tile's far edge


class StairsAscentDescent(Obstacle):
    """A one-way staircase spanning the full tile, alternating
    ascending/descending by repeat (`diff.col_index` parity) instead of
    mirroring up-then-down within a single tile like `raised_stairs`.

    Ascending tile: flat ground-level margin at the trailing edge (spawn),
    `n_steps` rising toward the leading edge, then a platform (elongated to
    the tile's far edge) at the leading edge (goal) — the robot starts
    below the stairs and its goal is on top. A descending tile is the
    mirror image: platform (start, top) at the trailing edge, `n_steps`
    down, flat margin (goal, ground level) at the leading edge.

    Step tread depth (`step_length`) is fixed; only the RISE per step
    (`diff.height / n_steps`) grades with difficulty.
    `extra.entry_margin`/`extra.step_length`/`extra.n_steps` override the
    defaults above. `extra.birth_platform_height` overrides ONLY the height
    used for the platform-side birth Z (build_usd/build_heightmap still use
    `diff.height` for the actual mesh) — an escape hatch for cross-wiring
    birth Z independently of the geometry, e.g. when two rows' terrain
    configs turn out swapped relative to some other consumer's expectations.
    """

    name = "stairs_ascent_descent"

    def _ascending(self, diff: DifficultyParams) -> bool:
        return diff.col_index % 2 == 0

    def _profile(self, tile: TileSpec, diff: DifficultyParams):
        entry_margin = diff.extra.get("entry_margin", ENTRY_MARGIN)
        step_length = diff.extra.get("step_length", STEP_LENGTH)
        n_steps = int(diff.extra.get("n_steps", N_STEPS))
        platform_length = tile.width - entry_margin - n_steps * step_length
        rise = diff.height / n_steps

        # leading edge (low X) -> trailing edge (high X): platform on top,
        # n_steps down, flat ground-level margin.
        widths = [platform_length] + [step_length] * n_steps + [entry_margin]
        tops = (
            [tile.base_z + diff.height]
            + [tile.base_z + rise * i for i in range(n_steps, 0, -1)]
            + [tile.base_z]
        )
        if not self._ascending(diff):
            # descending tile: mirror so the platform (start) sits at the
            # trailing edge and the ground-level margin (goal) at the
            # leading edge.
            widths = list(reversed(widths))
            tops = list(reversed(tops))

        centers = x_segment_centers(tile, widths)
        return list(zip(centers, widths, tops))

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        for i, (x, w, top) in enumerate(self._profile(tile, diff)):
            add_ground_slab(stage, f"{prim_path}/seg_{i}", x, 0.0, w, tile.depth, top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for x, w, top in self._profile(tile, diff):
            paint_rect(h, tile, x, 0.0, w, tile.depth, top)
        return h

    def birth_offsets(self, tile: TileSpec, diff: DifficultyParams) -> tuple[float, float]:
        # start = trailing edge, target = leading edge (fixed by birth.py)
        birth_height = diff.extra.get("birth_platform_height", diff.height)
        if self._ascending(diff):
            return 0.0, birth_height  # start at the bottom, target on top
        return birth_height, 0.0  # start on top, target at the bottom

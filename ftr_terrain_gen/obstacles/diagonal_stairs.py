from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, MeshObstacle, TileSpec
from ftr_terrain_gen.shapes import paint_profile_along, stair_profile_1d

STEP_LENGTH = 0.28  # tread depth — FIXED, only the rise grades
PLATFORM_LENGTH = 0.6  # flat platform on top
N_STEPS = 3  # steps up, then the same number down
MIN_YAW = 15.0  # yaw of the staircase axis from the travel direction, easiest repeat
MAX_YAW = 35.0  # ... hardest repeat


class DiagonalStairs(MeshObstacle):
    """`raised_stairs` turned `yaw` degrees in the lane: one track meets each
    riser before the other, so the body rolls on every step and the four
    flippers have to be timed per side instead of as front/rear pairs.
    Rise per step = `diff.height / n_steps`; yaw grades `min_yaw` ->
    `max_yaw` with `diff.t` and flips sign on odd repeats when
    `extra.mirror_alternate` is set. Evaluated per cell (mesh), clipped to
    the lane. `extra.step_length`/`platform_length`/`n_steps`/`min_yaw`/
    `max_yaw` override the defaults above. Keep `mesh_stride` at 1 — the
    risers need every cell.
    """

    name = "diagonal_stairs"

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        step_length = diff.extra.get("step_length", STEP_LENGTH)
        platform = diff.extra.get("platform_length", PLATFORM_LENGTH)
        n_steps = int(diff.extra.get("n_steps", N_STEPS))
        lo, hi = diff.extra.get("min_yaw", MIN_YAW), diff.extra.get("max_yaw", MAX_YAW)
        yaw = diff.mirror_sign * (lo + (hi - lo) * diff.t)
        rise = diff.height / n_steps

        def z(u):
            return stair_profile_1d(u, tile.base_z, rise, step_length, n_steps, platform, sign=1.0)

        h = self.flat_heightmap(tile)
        return paint_profile_along(h, tile, z, yaw_deg=yaw)

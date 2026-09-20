from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, MeshObstacle, TileSpec
from ftr_terrain_gen.shapes import cell_centers

HOLD_LENGTH = 2.0  # length (along X) over which the full roll angle is held
TWIST_LENGTH = 1.0  # length of each twist-in / twist-out transition from flat to full roll
MIN_ROLL = 5.0  # roll angle (degrees) at the easiest repeat when no min_roll/max_roll given
MAX_ROLL = 20.0


class CrossSlope(MeshObstacle):
    """The lane is tilted about the travel axis (a side slope, NIST's 15 deg
    "roll ramp"): height = tan(roll(x)) * y. The roll angle twists in from
    flat over `twist_length`, holds for `hold_length`, and twists back out,
    so there is no step at the entry — the surface is a helicoid, the
    left and right tracks are at different heights the whole way and the
    flippers on the downhill side have to do something different from the
    uphill side. Roll grades `min_roll` -> `max_roll` (degrees) with
    `diff.t`; `extra.mirror_alternate` flips the tilt direction on odd
    repeats. Uses `diff.height` only when no roll range is given (then
    height is the rise across half a lane).
    """

    name = "cross_slope"

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        hold = diff.extra.get("hold_length", HOLD_LENGTH)
        twist = diff.extra.get("twist_length", TWIST_LENGTH)
        if "min_roll" in diff.extra or "max_roll" in diff.extra:
            roll = diff.lerp_extra("min_roll", "max_roll")
        else:
            roll = np.degrees(np.arctan(diff.height / (tile.depth / 2))) if diff.height else MIN_ROLL + (MAX_ROLL - MIN_ROLL) * diff.t
        roll *= diff.mirror_sign
        gx, gy = cell_centers(tile)
        envelope = np.clip((hold / 2 + twist - np.abs(gx)) / twist, 0.0, 1.0)
        h = tile.base_z + np.tan(np.radians(roll)) * envelope * gy
        return h.astype(np.float32)

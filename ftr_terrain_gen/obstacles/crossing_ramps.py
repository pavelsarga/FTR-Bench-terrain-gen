from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, MeshObstacle, TileSpec
from ftr_terrain_gen.shapes import cell_centers

WEDGE_LENGTH = 0.6  # horizontal run of each ramp wedge (NIST crossing ramps use 0.6 m)
N_WEDGES = 6  # number of wedges along X (one up + one down = 2)
MIN_ANGLE = 4.0
MAX_ANGLE = 10.0  # the two halves are opposed, so the track-to-track height difference is TWICE
# the wedge height: 10 deg wedges on 0.6 m = 0.21 m between tracks 0.5 m apart, a 23 deg roll


class CrossingRamps(MeshObstacle):
    """NIST "crossing pitch/roll ramps": a triangle wave of wedges along the
    travel axis, in ANTI-PHASE between the left and right halves of the lane
    — while the left track climbs a wedge the right track descends one — so
    the body is pitched and rolled at once and the sign of both flips every
    `wedge_length`. Wedge angle grades `min_angle` -> `max_angle` (degrees)
    with `diff.t`; wedge height = wedge_length * tan(angle). Both halves
    start and end at ground level (the right half is the negated wave), so
    the entry is flush — and because they are opposed, the height difference
    between the two tracks peaks at 2 x the wedge height, i.e. the body roll
    is about twice what the wedge angle alone suggests (keep max_angle <= 10).
    """

    name = "crossing_ramps"

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        wedge = diff.extra.get("wedge_length", WEDGE_LENGTH)
        n = int(diff.extra.get("n_wedges", N_WEDGES))
        angle = diff.lerp_extra("min_angle", "max_angle") if "min_angle" in diff.extra else MIN_ANGLE + (MAX_ANGLE - MIN_ANGLE) * diff.t
        amp = wedge * np.tan(np.radians(angle))
        span = n * wedge
        gx, gy = cell_centers(tile)
        s = gx + span / 2  # 0 at the first wedge foot
        tri = 1.0 - np.abs((s / wedge) % 2.0 - 1.0)  # 0 -> 1 -> 0 over two wedges
        inside = np.abs(gx) <= span / 2
        side = np.where(gy >= 0, 1.0, -1.0) * diff.mirror_sign
        h = tile.base_z + np.where(inside, amp * tri * side, 0.0)
        return h.astype(np.float32)

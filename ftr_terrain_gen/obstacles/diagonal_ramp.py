from __future__ import annotations

import math

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, MeshObstacle, TileSpec
from ftr_terrain_gen.shapes import paint_profile_along

RAMP_LENGTH = 1.0  # horizontal run of each face of the ridge, along the ridge's own fall line
PLATEAU_LENGTH = 0.4  # flat crest between the two faces
MIN_YAW = 15.0  # default yaw of the fall line from the travel axis at the easiest repeat
MAX_YAW = 35.0  # ... and at the hardest
ANGLE = 15.0  # default face angle (degrees) when no min_angle/max_angle given
# centered in the tile


class DiagonalRamp(MeshObstacle):
    """An A-shaped ridge (two opposed ramps + short crest, like `pitch_ramp`)
    whose fall line is turned `yaw` degrees away from the direction of
    travel, so the two tracks reach the ramp foot at different times and the
    body rolls while it pitches — the sloped counterpart of `diagonal_trunk`.
    Yaw grades `min_yaw` -> `max_yaw` with `diff.t`; the face angle is
    `extra.angle` or graded `min_angle` -> `max_angle`. With
    `extra.mirror_alternate: true` odd repeats use the opposite yaw sign, so
    both handednesses appear at every difficulty. The ridge is evaluated per
    cell in the turned frame and clipped to the lane, so it never leaks into
    a neighbouring row (a rotated box would).
    """

    name = "diagonal_ramp"

    def _params(self, diff: DifficultyParams):
        ramp_length = diff.extra.get("ramp_length", RAMP_LENGTH)
        plateau = diff.extra.get("plateau_length", PLATEAU_LENGTH)
        if "min_angle" in diff.extra or "max_angle" in diff.extra:
            angle = diff.lerp_extra("min_angle", "max_angle")
        else:
            angle = diff.extra.get("angle", ANGLE)
        height = ramp_length * math.tan(math.radians(angle))
        yaw = diff.mirror_sign * (
            diff.extra.get("min_yaw", MIN_YAW) + (diff.extra.get("max_yaw", MAX_YAW) - diff.extra.get("min_yaw", MIN_YAW)) * diff.t
        )
        return ramp_length, plateau, height, yaw

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        ramp_length, plateau, height, yaw = self._params(diff)

        def z(u):
            d = np.abs(u) - plateau / 2
            frac = np.clip(1.0 - d / ramp_length, 0.0, 1.0)
            return tile.base_z + height * frac

        h = self.flat_heightmap(tile)
        return paint_profile_along(h, tile, z, yaw_deg=yaw)

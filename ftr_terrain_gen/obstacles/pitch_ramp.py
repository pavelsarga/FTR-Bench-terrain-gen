from __future__ import annotations

import math

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, MeshObstacle, TileSpec
from ftr_terrain_gen.shapes import paint_profile_along

RAMP_LENGTH = 1.2  # default horizontal run of each ramp — FIXED, the angle grades
PLATEAU_LENGTH = 1.0  # default flat crest (A) / trough floor (V) between the two ramps
SHAPE = "A"  # "A": up-ramp, crest, down-ramp. "V": down-ramp, trough, up-ramp.
# the whole feature is centered in the tile — flat margins on each side are equal


class PitchRamp(MeshObstacle):
    """Two opposed ramps with a flat section between them, spanning the full
    lane width. `shape: A` climbs to a crest and descends again; `shape: V`
    drops into a trough and climbs back out. Both are meshes (the first
    sloped surfaces in this generator — every other obstacle is boxes).

    The ramp ANGLE is what grades: `extra.min_angle`/`extra.max_angle`
    (degrees, lerped by `diff.t`) when present, else the angle whose rise
    over `ramp_length` equals `diff.height`. The A crest high-centres the
    robot (1.12 m of track over a ridge) and the V trough grounds its belly
    (7 cm clearance over a 0.56 m half-length is a ~7 deg breakover with the
    flippers flat) — either way the flippers have to lift the body, which no
    box obstacle in custom_mixed ever required.
    `extra.ramp_length`/`extra.plateau_length`/`extra.shape` override the
    defaults above; `extra.mesh_stride` coarsens the collision mesh.
    """

    name = "pitch_ramp"

    def _params(self, diff: DifficultyParams):
        ramp_length = diff.extra.get("ramp_length", RAMP_LENGTH)
        plateau = diff.extra.get("plateau_length", PLATEAU_LENGTH)
        if "min_angle" in diff.extra or "max_angle" in diff.extra:
            angle = diff.lerp_extra("min_angle", "max_angle")
            height = ramp_length * math.tan(math.radians(angle))
        else:
            height = diff.height
        sign = -1.0 if str(diff.extra.get("shape", SHAPE)).upper() == "V" else 1.0
        return ramp_length, plateau, height, sign

    def profile(self, tile: TileSpec, diff: DifficultyParams):
        ramp_length, plateau, height, sign = self._params(diff)

        def z(u):
            d = np.abs(u) - plateau / 2
            frac = np.clip(1.0 - d / ramp_length, 0.0, 1.0)  # 1 on the plateau, 0 past the ramp foot
            return tile.base_z + sign * height * frac

        return z

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        return paint_profile_along(h, tile, self.profile(tile, diff))

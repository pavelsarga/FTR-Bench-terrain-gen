from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, MeshObstacle, TileSpec
from ftr_terrain_gen.shapes import rotated_coords

WAVELENGTH = 1.2  # crest-to-crest distance (m)
N_PERIODS = 3  # whole periods, so the field starts and ends at ground level
MIN_YAW = 0.0  # wave-front yaw from perpendicular at the easiest repeat
MAX_YAW = 0.0


class Wave(MeshObstacle):
    """A sinusoidal ground swell (FTR-Bench "wave terrain") of amplitude
    `diff.height` (peak-to-trough is twice that) over `n_periods` whole
    wavelengths, so it starts and ends flush with the ground. Continuous
    curvature means the flippers must be modulated all the way through
    rather than set once per obstacle. Optional `min_yaw`/`max_yaw` turn the
    wave fronts so the swell also rolls the body; `extra.mirror_alternate`
    flips that sign on odd repeats. `extra.mesh_stride: 2` is fine here.
    """

    name = "wave"

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        wavelength = diff.extra.get("wavelength", WAVELENGTH)
        n = int(diff.extra.get("n_periods", N_PERIODS))
        lo, hi = diff.extra.get("min_yaw", MIN_YAW), diff.extra.get("max_yaw", MAX_YAW)
        yaw = diff.mirror_sign * (lo + (hi - lo) * diff.t)
        span = n * wavelength
        u, _ = rotated_coords(tile, 0.0, 0.0, yaw)
        inside = np.abs(u) <= span / 2
        z = diff.height * np.sin(2 * np.pi * (u + span / 2) / wavelength)
        h = tile.base_z + np.where(inside, z, 0.0)
        return h.astype(np.float32)

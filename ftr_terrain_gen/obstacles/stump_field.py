from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, MeshObstacle, TileSpec
from ftr_terrain_gen.shapes import cell_centers

N_STUMPS = 3  # bumps along the path (3 x 1.2 m + the bumps' own ~1 m skirts fit the 4.5 m the
# 8.8 m tile leaves once both spawn margins are kept flat)
SPACING = 1.0  # X distance between consecutive bump centres — close enough that the saddle between two bumps stays well above ground
SCALE = 0.8  # bump radius parameter (stump.py uses 0.5): half height at r ~ 0.53 m, ~1.5 m wide
STEEPNESS = 4  # same profile as stump.py: sigmoid'(STEEPNESS * r^2 / SCALE^2)
OFFSET = 0.40  # bumps alternate +/- this far off the centreline, so the robot is always on one flank
HEIGHT_RAMP = 0.7  # the first bump met is this fraction of diff.height, the last is the full height
JITTER = 1.0  # scale of the per-tile randomisation below (0 = the fixed, regular layout)
_X_JITTER = 0.12  # (m) each bump centre moves this far along the path at JITTER 1
_Y_RANGE = (0.30, 0.45)  # (m) the lateral offset is drawn from this range instead of being exactly OFFSET
_SCALE_RANGE = (0.94, 1.06)  # bump width factor
_HEIGHT_JITTER = 0.10  # each non-tallest bump's height factor is scaled by U(1 - this, 1)
_PEAK = 0.25  # sigmoid'(0)


def _bump(r2: np.ndarray) -> np.ndarray:
    e = np.exp(-r2)
    return e / (1.0 + e) ** 2


class StumpField(MeshObstacle):
    """`n_stumps` smooth radial bumps (the `stump` profile) staggered along
    the lane, every other one on the other side of the centreline, close
    enough that the robot is never on flat ground between them: it climbs
    the flank of one while descending the last, with the two tracks at
    different heights the whole way. Peak height grades with `diff.height`
    across the row (0.20 -> 0.35 in the v2 course; the mound limit for FTR
    was 0.60, and these are smooth, so the belly is what limits it, not the
    flipper reach), and within one tile the bumps grow along the path from
    `height_ramp` x that height (first bump met, at the +X spawn end) to the
    full height (last bump).
    The layout is RANDOMISED per tile from `diff.seed` (`extra.jitter`, default
    1.0; 0 gives the regular pattern): each bump moves up to 0.12 m along the
    path, its lateral offset is drawn from 0.30-0.45 m (the side still
    alternates), its width varies +-6 % and every bump but the tallest loses
    up to 10 % of its height — enough that no two repeats have the same
    flank sequence, while the saddles between bumps stay above ground so
    the centreline is never flat inside the field. `extra.mirror_alternate`
    flips the stagger on odd repeats. Built as a mesh (`MeshObstacle`),
    unlike `stump`'s grid of 84 x 84 boxes.
    """

    name = "stump_field"

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        n = int(diff.extra.get("n_stumps", N_STUMPS))
        spacing = diff.extra.get("spacing", SPACING)
        scale = diff.extra.get("scale", SCALE)
        offset = diff.extra.get("offset", OFFSET) * diff.mirror_sign
        ramp = diff.extra.get("height_ramp", HEIGHT_RAMP)
        jitter = float(diff.extra.get("jitter", JITTER))
        rng = diff.rng()
        gx, gy = cell_centers(tile)
        h = np.zeros_like(gx)
        x0 = -(n - 1) * spacing / 2
        for k in range(n):
            # k = 0 sits at -X (met last when driving -X): full height there, ramp down toward +X
            cx = x0 + k * spacing + jitter * rng.uniform(-_X_JITTER, _X_JITTER)
            side = 1.0 if k % 2 == 0 else -1.0
            lateral = abs(offset) + jitter * (rng.uniform(*_Y_RANGE) - OFFSET)
            cy = side * np.sign(offset) * lateral
            s_k = scale * (1.0 + jitter * (rng.uniform(*_SCALE_RANGE) - 1.0))
            frac = 1.0 if n == 1 else 1.0 - (1.0 - ramp) * k / (n - 1)
            if k > 0:
                frac *= 1.0 - jitter * rng.uniform(0.0, _HEIGHT_JITTER)
            r2 = ((gx - cx) / s_k) ** 2 + ((gy - cy) / s_k) ** 2
            h = np.maximum(h, frac * diff.height * _bump(STEEPNESS * r2) / _PEAK)
        h[h < 0.005] = 0.0
        return (tile.base_z + h).astype(np.float32)

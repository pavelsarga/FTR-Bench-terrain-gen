"""Per-tile feature offsets: `extra.feature_offset: {x: <max>, y: <max>}` on a
row shifts every tile's obstacle by a seeded random amount in [-max, max]
along the path (x) and across the lane (y).

Every obstacle centres its feature in the tile it is given, so the shift is
done by handing it a SMALLER tile — width - 2|dx|, depth - 2|dy| — placed
off-centre, and filling the vacated strips with flat ground. Nothing ever
extends past the tile boundary (a translated box would poke into the
neighbouring lane/column), recessed features keep their walls, and the flat
spawn/goal margins simply become asymmetric: the robot meets the feature
earlier or later than the training courses' fixed "centre of the tile".
That is the point — a policy that learned WHERE the obstacle is, rather
than what it looks like, is exposed on the holdout course.

Types listed in NO_OFFSET are left alone: their geometry must reach the
tile edge (a one-way staircase's platform) or they are already sequences.
"""

from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.registry import get_obstacle
from ftr_terrain_gen.shapes import paint_rect

NO_OFFSET = {"stairs_ascent_descent", "sequence", "flat_patch"}
_SEED_SALT = 424242  # separate stream from the obstacle's own diff.rng()


class OffsetObstacle(Obstacle):
    def __init__(self, base: Obstacle, dx: float, dy: float):
        self.base, self.dx, self.dy = base, dx, dy
        self.name = base.name

    def _sub(self, tile: TileSpec) -> TileSpec:
        return TileSpec(
            width=tile.width - 2 * abs(self.dx), depth=tile.depth - 2 * abs(self.dy),
            cell_size=tile.cell_size, base_z=tile.base_z,
        )

    def _fillers(self, tile: TileSpec):
        """(x_center, y_center, width, depth) of the flat strips the shrunken tile leaves."""
        strips = []
        if self.dx:
            w = 2 * abs(self.dx)
            xc = -tile.width / 2 + w / 2 if self.dx > 0 else tile.width / 2 - w / 2
            strips.append((xc, 0.0, w, tile.depth))
        if self.dy:
            d = 2 * abs(self.dy)
            yc = -tile.depth / 2 + d / 2 if self.dy > 0 else tile.depth / 2 - d / 2
            strips.append((0.0, yc, tile.width, d))
        return strips

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        from pxr import Gf, UsdGeom

        from ftr_terrain_gen.usd_utils import add_ground_slab

        for k, (xc, yc, w, d) in enumerate(self._fillers(tile)):
            add_ground_slab(stage, f"{prim_path}/filler_{k}", xc, yc, w, d, tile.base_z, tile)
        sub_path = f"{prim_path}/offset"
        xform = UsdGeom.Xform.Define(stage, sub_path)
        UsdGeom.Xformable(xform).AddTranslateOp().Set(Gf.Vec3d(self.dx, self.dy, 0.0))
        self.base.build_usd(stage, sub_path, self._sub(tile), diff)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        sub = self._sub(tile)
        patch = self.base.build_heightmap(sub, diff)
        i0 = int(round((self.dx - sub.width / 2 + tile.width / 2) / tile.cell_size))
        j0 = int(round((self.dy - sub.depth / 2 + tile.depth / 2) / tile.cell_size))
        i1, j1 = min(i0 + patch.shape[0], tile.nx), min(j0 + patch.shape[1], tile.ny)
        h[i0:i1, j0:j1] = patch[: i1 - i0, : j1 - j0]
        for xc, yc, w, d in self._fillers(tile):
            paint_rect(h, tile, xc, yc, w, d, tile.base_z)
        return h

    def birth_offsets(self, tile: TileSpec, diff: DifficultyParams) -> tuple[float, float]:
        return self.base.birth_offsets(self._sub(tile), diff)


def resolve_obstacle(obstacle_type: str, diff: DifficultyParams) -> Obstacle:
    """The obstacle for a tile, wrapped with its random offset when the row asks for one."""
    base = get_obstacle(obstacle_type)
    spec = diff.extra.get("feature_offset")
    if not spec or obstacle_type in NO_OFFSET:
        return base
    mx = float(spec.get("x", 0.0))
    my = float(spec.get("y", 0.0))
    rng = np.random.default_rng((diff.seed + _SEED_SALT) % (2**32))
    dx = float(rng.uniform(-mx, mx)) if mx > 0 else 0.0
    dy = float(rng.uniform(-my, my)) if my > 0 else 0.0
    # snap to the heightmap grid so the sub-patch lands on whole cells
    cs = 0.05
    dx, dy = round(dx / cs) * cs, round(dy / cs) * cs
    if dx == 0.0 and dy == 0.0:
        return base
    return OffsetObstacle(base, dx, dy)

from __future__ import annotations

import math

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rotated_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

TRUNK_WIDTH = 0.5  # default trunk cross-section thickness
MIN_ANGLE = -45.0  # default tilt-from-perpendicular at the easiest repeat (diff.t == 0)
MAX_ANGLE = 45.0  # default tilt-from-perpendicular at the hardest repeat (diff.t == 1)
# 0 deg == straight across (perpendicular, not actually diagonal) — the tilt
# lerp deliberately never lands exactly on 0 (see _tilt_deg), so the trunk is
# always at least MIN_TILT_MAGNITUDE off perpendicular, in one direction or
# the other depending on where MIN_ANGLE/MAX_ANGLE put t's zero-crossing.
MIN_TILT_MAGNITUDE = 1.0
WALL_WIDTH = 0.25  # default guard-wall thickness on each side
WALL_LENGTH = 3.0  # default guard-wall length along X — FIXED, independent of the trunk's angle
WALL_HEIGHT = 0.15  # default guard-wall height ABOVE the trunk's own top surface


class DiagonalTrunk(Obstacle):
    """A single raised rectangular block crossing the tile diagonally — its
    Y-extent spans the full lane width (tile.depth) — enclosed on both Y
    sides by thin flat guard walls at the tile's edges (axis-aligned, NOT
    rotated with the trunk — like twin_rails' walls), spanning the trunk's
    X footprint, taller than the trunk itself, so the robot can't slide off
    sideways while crossing at an angle.

    Unlike the other obstacles, the graded quantity is the CROSSING ANGLE
    (tilt from perpendicular), not a size: `min_angle`/`max_angle` (default
    -45/45 degrees) lerp by `diff.t`, and the trunk's length is derived from
    that angle plus tile.depth (a steeper tilt needs a longer trunk to still
    reach both lane edges). `diff.height` is the trunk's height above the
    ground; the guard walls sit `wall_height` above THAT. The walls'
    `wall_length` is FIXED (default 3m) — unlike the trunk itself, it does
    NOT vary with the crossing angle.
    `extra.min_angle`/`extra.max_angle`/`extra.trunk_width`/`extra.wall_width`/
    `extra.wall_length`/`extra.wall_height` override the defaults above.
    """

    name = "diagonal_trunk"

    def _tilt_deg(self, diff: DifficultyParams) -> float:
        lo = diff.extra.get("min_angle", MIN_ANGLE)
        hi = diff.extra.get("max_angle", MAX_ANGLE)
        if diff.extra.get("mirror_alternate", False):
            # grade the MAGNITUDE min->max and alternate the sign by repeat, so
            # every tilt is seen in both handednesses (custom_mixed graded
            # -45 -> +45 across the row and the policy came out handed:
            # 0.97 success at -45 deg, 0.31 at +45 deg)
            tilt = (abs(lo) + (abs(hi) - abs(lo)) * diff.t) * diff.mirror_sign
        else:
            tilt = lo + (hi - lo) * diff.t
        if abs(tilt) < MIN_TILT_MAGNITUDE:
            tilt = MIN_TILT_MAGNITUDE if tilt >= 0 else -MIN_TILT_MAGNITUDE
        return tilt

    def _geometry(self, tile: TileSpec, diff: DifficultyParams):
        tilt_deg = self._tilt_deg(diff)
        width = diff.extra.get("trunk_width", TRUNK_WIDTH)
        length = tile.depth / math.cos(math.radians(tilt_deg))
        angle_deg = 90.0 - tilt_deg  # rotate_z_deg: 90 == perpendicular (no tilt)
        return length, width, angle_deg

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        length, width, angle_deg = self._geometry(tile, diff)
        trunk_top = tile.base_z + diff.height
        add_ground_slab(stage, f"{prim_path}/trunk", 0.0, 0.0, length, width, trunk_top, tile, angle_deg)

        wall_length = diff.extra.get("wall_length", WALL_LENGTH)
        wall_width = diff.extra.get("wall_width", WALL_WIDTH)
        wall_top = trunk_top + diff.extra.get("wall_height", WALL_HEIGHT)
        wall_y = tile.depth / 2 - wall_width / 2
        add_ground_slab(stage, f"{prim_path}/wall_pos", 0.0, wall_y, wall_length, wall_width, wall_top, tile)
        add_ground_slab(stage, f"{prim_path}/wall_neg", 0.0, -wall_y, wall_length, wall_width, wall_top, tile)

    def build_lips(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams, spec) -> int | None:
        # the trunk is a yaw-rotated box: its edges are not cell-aligned, so lay the nosings
        # along the real box edges instead of the heightmap's staircase approximation
        from ftr_terrain_gen.edges import add_box_edge_lips, lip_friction
        from ftr_terrain_gen.usd_utils import bind_friction, ensure_friction_material

        length, width, angle_deg = self._geometry(tile, diff)
        n = add_box_edge_lips(stage, prim_path, 0.0, 0.0, length, width, angle_deg, tile.base_z + diff.height, spec)
        bind_friction(stage, prim_path, ensure_friction_material(stage, lip_friction(spec)))
        return n

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        length, width, angle_deg = self._geometry(tile, diff)
        trunk_top = tile.base_z + diff.height
        paint_rotated_rect(h, tile, 0.0, 0.0, length, width, angle_deg, trunk_top)
        return h

from __future__ import annotations

import math

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_cylinder_across
from ftr_terrain_gen.usd_utils import add_cylinder, add_ground_slab

MIN_YAW = 0.0  # log axis yaw from perpendicular-to-travel, easiest repeat
MAX_YAW = 0.0
SINK = 0.0  # how far the log is buried below "resting on the surface"
RIB_PITCH = 0.03  # (m of arc) spacing of the grip ribs around the log when the row sets `edge_lip`
RIB_WIDTH = 0.02  # (m) tangential width of a rib; the 1 cm between ribs is bare log
RIB_SPAN_DEG = 85.0  # ribs cover the upper arc from -85 to +85 deg off the crest


class RoundLog(Obstacle):
    """A cylinder lying ACROSS the lane (a fallen tree): `diff.height` is its
    diameter, i.e. the crest height. Unlike a step there is no edge for the
    flipper belt to hook — the flippers have to press down on the round
    surface and the body rolls over the crest — and the tracks leave it on a
    convex drop. `min_yaw`/`max_yaw` turn the log in the lane so one track
    reaches it first (the `diagonal_log` row uses this with
    `mirror_alternate`); the cylinder is lengthened to keep spanning the
    full lane at that yaw. Native cylinder collider, no mesh.
    """

    name = "round_log"

    def _params(self, tile: TileSpec, diff: DifficultyParams):
        lo, hi = diff.extra.get("min_yaw", MIN_YAW), diff.extra.get("max_yaw", MAX_YAW)
        yaw = diff.mirror_sign * (lo + (hi - lo) * diff.t)
        radius = diff.height / 2
        sink = diff.extra.get("sink", SINK)
        length = tile.depth / math.cos(math.radians(yaw))
        return yaw, radius, sink, length

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        yaw, radius, sink, length = self._params(tile, diff)
        add_cylinder(
            stage, f"{prim_path}/log", radius, length, "Y", (0.0, 0.0, tile.base_z + radius - sink), rotate_z_deg=yaw,
        )

    def build_lips(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams, spec) -> int | None:
        """Grip RIBS instead of edge lips: a smooth cylinder has no rising
        edge for `edges.py` to find, so the row's `edge_lip` becomes a ratchet
        of thin tangent slabs around the upper arc — `rib_pitch` apart, `height`
        proud of the surface (the same 1 cm the track protrusions stand for),
        following the log's yaw — bound to the lip material. The bare log
        between them keeps the scene friction, like the treads of a stepped hill.
        """
        from ftr_terrain_gen.edges import DEFAULT_LIP
        from ftr_terrain_gen.usd_utils import LIP_COLOR, add_tilted_slab, bind_friction, ensure_friction_material, set_display_color

        cfg = {**DEFAULT_LIP, **(spec if isinstance(spec, dict) else {})}
        height = float(cfg["height"])
        pitch = float(cfg.get("rib_pitch", RIB_PITCH))
        rib_w = float(cfg.get("rib_width", RIB_WIDTH))
        yaw, radius, sink, length = self._params(tile, diff)
        cz = tile.base_z + radius - sink
        n_side = int(math.radians(RIB_SPAN_DEG) * radius / pitch)
        n = 0
        for k in range(-n_side, n_side + 1):
            phi = k * pitch / radius  # angle off the crest, positive toward the yawed +X (the -X approach... both sides ribbed)
            # surface point in the cross-section plane (yawed X, Z) pushed `height` out along the normal
            nx, nz = math.sin(phi), math.cos(phi)
            px, pz = (radius + height) * nx, cz + (radius + height) * nz
            top = (px * math.cos(math.radians(yaw)), px * math.sin(math.radians(yaw)), pz)
            prim = add_tilted_slab(stage, f"{prim_path}/rib_{n:03d}", top, rib_w, length,
                                   slope_deg=-math.degrees(phi), yaw_deg=yaw, thickness=height + 0.01)
            set_display_color(prim, LIP_COLOR)
            n += 1
        bind_friction(stage, prim_path, ensure_friction_material(stage, float(cfg["friction"])))
        return n

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        yaw, radius, sink, _ = self._params(tile, diff)
        return paint_cylinder_across(h, tile, 0.0, radius, tile.base_z, yaw_deg=yaw, sink=sink)

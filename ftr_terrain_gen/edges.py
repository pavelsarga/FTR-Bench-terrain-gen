"""Edge lips: high-friction "nosings" along the top edges of steps.

The real MARV track is hard rubber with ~1 cm protrusions: on a smooth surface it
is not especially grippy, but on a step it catches the EDGE with a protrusion and
pulls the robot up mechanically. PhysX has neither the belt nor the protrusions,
so a step in the sim is only climbable if the flipper material does all the work —
which is why the course friction had to be cranked to values (mu_eff ~ 5) that
make skid-steering impossible on flat ground.

This module splits the two: every rising edge of a tile's heightmap (a step, a
stair riser, a platform edge, the wall of a pit) gets a thin lip box — a stair
NOSING — along it: it overhangs the riser face by `width` (so none of its faces
is coplanar with the slab's; a wheel touching the riser near the top meets the
lip alone, never two materials at once), stands `height` proud of the tread
(default 1 cm: the size of the track protrusion it stands in for) and reaches
`width` down the riser face. It is bound to its own high-friction material;
everything else keeps the row's / scene's ordinary friction. Contacts at an
edge land on the lip (grip + a small catch), contacts on treads and flat
ground do not.

Lips are NOT painted into the observation heightmap: the real robot's elevation
map would not resolve a 1 cm nosing either.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ftr_terrain_gen.obstacle_base import TileSpec

DEFAULT_LIP = {"height": 0.01, "width": 0.03, "friction": 3.0, "min_rise": 0.04}


@dataclass(frozen=True)
class Lip:
    """One lip box: `along` is "y" for an edge that runs across the lane (a riser
    faced when driving in X), "x" for one that runs along it."""

    x: float  # centre (tile-local)
    y: float
    z_top: float  # tread height at the edge (absolute); the box spans z_top - width .. z_top + height
    length: float  # extent along the edge
    along: str  # "x" | "y"


def _runs(mask_1d: np.ndarray) -> list[tuple[int, int]]:
    """[(start, stop)] index runs of True in a 1-D bool array (stop exclusive)."""
    out = []
    start = None
    for i, v in enumerate(mask_1d):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i))
            start = None
    if start is not None:
        out.append((start, len(mask_1d)))
    return out


def find_lips(h: np.ndarray, tile: TileSpec, min_rise: float, width: float) -> list[Lip]:
    """Rising edges of `h` (shape (nx, ny), absolute Z) as merged lip boxes.

    An edge exists between two neighbouring cells whose heights differ by at
    least `min_rise`; the lip hangs OUTWARD from the higher cell's riser face
    (over the lower cell). Neighbouring boundary cells along the edge with the same tread
    height are merged into one box, so a straight riser is a single prim and
    a diagonal one becomes a staircase of short ones.
    """
    c = tile.cell_size
    lips: list[Lip] = []
    x0, y0 = -tile.width / 2, -tile.depth / 2
    # edges between columns i and i+1 (riser faces perpendicular to X; lip runs along Y)
    dx = h[1:, :] - h[:-1, :]  # (nx-1, ny)
    for i in range(dx.shape[0]):
        xb = x0 + (i + 1) * c  # boundary x
        for sign in (1, -1):  # +1: cell i+1 is higher (lip on its side), -1: cell i is higher
            mask = (sign * dx[i]) >= min_rise
            for a, b in _runs(mask):
                tops = h[i + 1 if sign > 0 else i, a:b]
                # split the run where the tread height changes (a diagonal edge)
                s = a
                for j in range(a + 1, b + 1):
                    if j == b or abs(tops[j - a] - tops[s - a]) > 1e-4:
                        lips.append(Lip(x=xb - sign * width / 2, y=y0 + (s + j) / 2 * c,
                                        z_top=float(tops[s - a]), length=(j - s) * c, along="y"))
                        s = j
    # edges between rows j and j+1 (faces perpendicular to Y; lip runs along X)
    dy = h[:, 1:] - h[:, :-1]  # (nx, ny-1)
    for j in range(dy.shape[1]):
        yb = y0 + (j + 1) * c
        for sign in (1, -1):
            mask = (sign * dy[:, j]) >= min_rise
            for a, b in _runs(mask):
                tops = h[a:b, j + 1 if sign > 0 else j]
                s = a
                for i in range(a + 1, b + 1):
                    if i == b or abs(tops[i - a] - tops[s - a]) > 1e-4:
                        lips.append(Lip(x=x0 + (s + i) / 2 * c, y=yb - sign * width / 2,
                                        z_top=float(tops[s - a]), length=(i - s) * c, along="x"))
                        s = i
    return lips


def lip_friction(spec) -> float:
    cfg = {**DEFAULT_LIP, **(spec if isinstance(spec, dict) else {})}
    return float(cfg["friction"])


def build_lips_usd(stage, root_path: str, h: np.ndarray, tile: TileSpec, spec: dict) -> int:
    """Add the lip boxes for heightmap `h` under `root_path` and bind their
    friction material; returns the number of lips. `spec` overrides DEFAULT_LIP."""
    from ftr_terrain_gen.usd_utils import LIP_COLOR, add_cube, bind_friction, ensure_friction_material

    cfg = {**DEFAULT_LIP, **(spec if isinstance(spec, dict) else {})}
    height, width = float(cfg["height"]), float(cfg["width"])
    lips = find_lips(h, tile, float(cfg["min_rise"]), width)
    for k, lip in enumerate(lips):
        size = (width, lip.length, height + width) if lip.along == "y" else (lip.length, width, height + width)
        add_cube(stage, f"{root_path}/lip_{k:04d}", size, translate=(lip.x, lip.y, lip.z_top + height),
                 color=LIP_COLOR)
    if lips:
        bind_friction(stage, root_path, ensure_friction_material(stage, float(cfg["friction"])))
    return len(lips)


def add_box_edge_lips(stage, root_path: str, cx: float, cy: float, length: float, width: float,
                      yaw_deg: float, z_top: float, spec, start: int = 0) -> int:
    """Nosings along the four top edges of a yaw-rotated box (a rotated ground
    slab such as `diagonal_trunk`), for shapes whose edges are not cell-aligned
    and therefore invisible to `find_lips`. Returns the number of lips added."""
    import math

    from ftr_terrain_gen.usd_utils import LIP_COLOR, add_cube

    cfg = {**DEFAULT_LIP, **(spec if isinstance(spec, dict) else {})}
    height, w = float(cfg["height"]), float(cfg["width"])
    c, sn = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    k = start
    # long edges (offset across, along the box's local y) and short edges (along local x)
    for dx, dy, size in ((0.0, width / 2 + w / 2, (length + 2 * w, w, height + w)),
                         (0.0, -(width / 2 + w / 2), (length + 2 * w, w, height + w)),
                         (length / 2 + w / 2, 0.0, (w, width, height + w)),
                         (-(length / 2 + w / 2), 0.0, (w, width, height + w))):
        add_cube(stage, f"{root_path}/lip_{k:04d}", size,
                 translate=(cx + dx * c - dy * sn, cy + dx * sn + dy * c, z_top + height),
                 rotate_z_deg=yaw_deg, color=LIP_COLOR)
        k += 1
    return k - start


def add_slab_edge_lips(stage, root_path: str, top_center, length: float, width: float,
                       slope_deg: float, yaw_deg: float, spec, start: int = 0) -> int:
    """Nosings along the four edges of a tilted slab (`tilted_pallet`): each lip
    continues the slab's plane past the edge, `height` above it, so the
    overhang at the high edge is a real nose and the low/side lips lie in the
    plane. Lips of a partly buried slab end up underground, which is harmless."""
    from ftr_terrain_gen.usd_utils import LIP_COLOR, _rotation_columns, add_tilted_slab, set_display_color

    cfg = {**DEFAULT_LIP, **(spec if isinstance(spec, dict) else {})}
    height, w = float(cfg["height"]), float(cfg["width"])
    u, v, n = _rotation_columns(yaw_deg, slope_deg)
    cx, cy, cz = top_center
    k = start
    for a, b, ln, wd in ((length / 2 + w / 2, 0.0, w, width), (-(length / 2 + w / 2), 0.0, w, width),
                         (0.0, width / 2 + w / 2, length + 2 * w, w), (0.0, -(width / 2 + w / 2), length + 2 * w, w)):
        top = tuple(cc + a * uu + b * vv + height * nn for cc, uu, vv, nn in zip((cx, cy, cz), u, v, n))
        prim = add_tilted_slab(stage, f"{root_path}/lip_{k:04d}", top, ln, wd, slope_deg, yaw_deg, thickness=height + w)
        set_display_color(prim, LIP_COLOR)
        k += 1
    return k - start

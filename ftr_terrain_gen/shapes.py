"""Small shared geometry helper (rectangle painting into a heightmap) reused
by obstacle classes, so each obstacle file only holds its own parametrization.
"""

from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import TileSpec

# Width of the flat "level beam" left on EACH Y side of a recessed feature
# (a trench doesn't span the full tile.depth lane width — it stops
# MIN_EDGE_MARGIN short on each side, with a flat beam filling that strip).
# Without this, a trench's side would sit flush with the tile's Y edge,
# directly abutting whatever the neighboring LANE's terrain is with no
# ground between them.
MIN_EDGE_MARGIN = 0.1


def rect_indices(tile: TileSpec, x_center: float, y_center: float, width: float, depth: float):
    def idx(center, extent, n, tile_extent):
        lo = (center - extent / 2 + tile_extent / 2) / tile.cell_size
        hi = (center + extent / 2 + tile_extent / 2) / tile.cell_size
        return int(np.clip(round(lo), 0, n)), int(np.clip(round(hi), 0, n))

    i0, i1 = idx(x_center, width, tile.nx, tile.width)
    j0, j1 = idx(y_center, depth, tile.ny, tile.depth)
    return i0, i1, j0, j1


def paint_rect(
    h: np.ndarray, tile: TileSpec, x_center: float, y_center: float, width: float, depth: float, z_value: float
) -> np.ndarray:
    i0, i1, j0, j1 = rect_indices(tile, x_center, y_center, width, depth)
    if i1 > i0 and j1 > j0:
        h[i0:i1, j0:j1] = z_value
    return h


def paint_rotated_rect(
    h: np.ndarray,
    tile: TileSpec,
    x_center: float,
    y_center: float,
    length: float,
    width: float,
    angle_deg: float,
    z_value: float,
) -> np.ndarray:
    """Like paint_rect, but for a rectangle rotated `angle_deg` about Z
    (matching add_cube's rotate_z_deg convention) — `length` runs along the
    rectangle's own rotated local X axis, `width` along its local Y axis.
    Tests every heightmap cell against the rotated footprint directly (no
    fast index-slice shortcut exists for an arbitrary rotation).
    """
    xs = np.linspace(-tile.width / 2, tile.width / 2, tile.nx, endpoint=False) + tile.width / (2 * tile.nx)
    ys = np.linspace(-tile.depth / 2, tile.depth / 2, tile.ny, endpoint=False) + tile.depth / (2 * tile.ny)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    theta = np.radians(angle_deg)
    dx, dy = gx - x_center, gy - y_center
    local_x = dx * np.cos(theta) + dy * np.sin(theta)
    local_y = -dx * np.sin(theta) + dy * np.cos(theta)
    mask = (np.abs(local_x) <= length / 2) & (np.abs(local_y) <= width / 2)
    h[mask] = z_value
    return h


def x_segment_centers(tile: TileSpec, widths: list[float]) -> list[float]:
    """Given ordered segment widths along X (left to right, starting at the
    tile's leading edge), return each segment's center X. Widths should sum
    to tile.width. Use this — NOT a full-tile base slab plus an embedded
    extra piece — for any RECESSED feature: a full-width base slab always
    stays the tallest (and therefore the only collidable) surface at its own
    footprint, so a trench cube added underneath it has no physical effect.
    Non-overlapping segments, each its own ground_slab, is what actually
    creates a dip.
    """
    x = -tile.width / 2
    centers = []
    for w in widths:
        centers.append(x + w / 2)
        x += w
    return centers


def value_noise_2d(rng: np.random.Generator, nx: int, ny: int, coarse_n: int = 3) -> np.ndarray:
    """Smooth pseudo-random field in [0, 1], shape (nx, ny): sample a coarse
    `coarse_n` x `coarse_n` grid of independent random control points, then
    bilinearly upsample to (nx, ny). Neighboring output cells are
    correlated (unlike per-cell independent noise), which is what gives a
    grid built from this a cohesive, blended terrain look instead of
    checkerboard static — used by rock_formation for its height field.
    """
    coarse = rng.uniform(0.0, 1.0, size=(coarse_n, coarse_n))
    xs = np.linspace(0, coarse_n - 1, nx)
    ys = np.linspace(0, coarse_n - 1, ny)
    x0 = np.floor(xs).astype(int).clip(0, coarse_n - 2)
    y0 = np.floor(ys).astype(int).clip(0, coarse_n - 2)
    fx = (xs - x0)[:, None]
    fy = (ys - y0)[None, :]
    c00 = coarse[x0][:, y0]
    c10 = coarse[x0 + 1][:, y0]
    c01 = coarse[x0][:, y0 + 1]
    c11 = coarse[x0 + 1][:, y0 + 1]
    return c00 * (1 - fx) * (1 - fy) + c10 * fx * (1 - fy) + c01 * (1 - fx) * fy + c11 * fx * fy


def symmetric_stair_profile(
    tile: TileSpec,
    front_margin: float,
    step_length: float,
    platform_length: float,
    n_steps: int,
    height: float,
    sign: float,
) -> list[tuple[float, float, float]]:
    """Segment layout for a symmetric staircase spanning the full
    tile.width: flat spawn, `n_steps` steps up (or down, if sign=-1), a flat
    platform, `n_steps` steps back down (mirrored), flat goal. Step tread
    depth (`step_length`) is fixed; only the RISE per step (`height /
    n_steps`) varies with difficulty. Returns (x_center, width, top_z)
    tuples ready for add_ground_slab/paint_rect — `sign=+1` for a raised
    staircase (platform above tile.base_z), `sign=-1` for a recessed one
    (platform below).
    """
    back_margin = tile.width - front_margin - 2 * n_steps * step_length - platform_length
    widths = [front_margin] + [step_length] * n_steps + [platform_length] + [step_length] * n_steps + [back_margin]
    rise = height / n_steps
    tops = (
        [tile.base_z]
        + [tile.base_z + sign * rise * i for i in range(1, n_steps + 1)]
        + [tile.base_z + sign * height]
        + [tile.base_z + sign * rise * i for i in range(n_steps, 0, -1)]
        + [tile.base_z]
    )
    centers = x_segment_centers(tile, widths)
    return list(zip(centers, widths, tops))

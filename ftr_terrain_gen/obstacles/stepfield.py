from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_rect
from ftr_terrain_gen.usd_utils import add_ground_slab

POST_SIZE = 0.15  # square post side (m) — 3 heightmap cells, so every post is resolved exactly
N_POSTS = 16  # posts per side: 16 x 0.15 = 2.4 m field, two NIST medium pallets
N_LEVELS = 4  # post heights are 0, 1, 2 or 3 units
MAX_STEP = 2  # NIST rule 1: adjacent posts differ by at most 2 units
LAYOUTS = ("flat_square", "hill", "flat_cross", "diagonal_hill")  # cycled by repeat
# the field is centered in the tile — flat margins on each side are equal


class Stepfield(Obstacle):
    """A NIST/RoboCupRescue stepfield pallet (Jacoff et al. 2008), scaled to
    MARV: a grid of square posts at integer height levels. `diff.height` is
    the height of the TALLEST level (3 units), so the unit is height/3 and a
    0.30 m field has 0.10 m posts like the medium NIST pallet. Layout rules
    are the paper's, which is what makes the field traversable BY
    CONSTRUCTION instead of by seed luck (custom_mixed's random cobblestones
    and cur_mixed's 0.40 m block fields are both layout lotteries):
      * adjacent posts never differ by more than `MAX_STEP` units;
      * `flat_square` / `flat_cross`: levels 0-2 at random, a handful of
        prescribed level-3 posts (the square's corners / the cross's arms);
      * `hill`: a level-3 ridge across the lane, flanked by rows at 2-3,
        the rest 0-2 — a ridge the robot has to crest;
      * `diagonal_hill`: the same ridge along the field's diagonal, so the
        two tracks crest it at different times.
    The layout cycles through LAYOUTS by repeat (or `extra.layout` fixes
    one). `extra.post_size`/`n_posts` override the defaults.
    """

    name = "stepfield"

    def _levels(self, diff: DifficultyParams) -> np.ndarray:
        n = int(diff.extra.get("n_posts", N_POSTS))
        layout = diff.extra.get("layout") or LAYOUTS[diff.col_index % len(LAYOUTS)]
        rng = diff.rng()
        lv = rng.integers(0, 3, size=(n, n))  # 0..2
        if layout == "flat_square":
            q = n // 4
            for i, j in ((q, q), (q, n - 1 - q), (n - 1 - q, q), (n - 1 - q, n - 1 - q)):
                lv[i, j] = 3
        elif layout == "flat_cross":
            c = n // 2
            for k in range(1, n // 3):
                lv[c, c + k] = lv[c, c - k] = lv[c + k, c] = lv[c - k, c] = 3
        elif layout == "hill":
            c = n // 2
            lv[c, :] = 3
            for k in (1, 2):
                lv[c - k, :] = rng.integers(2, 4, size=n)
                lv[c + k, :] = rng.integers(2, 4, size=n)
        elif layout == "diagonal_hill":
            for i in range(n):
                lv[i, i] = 3
                for k in (1, 2):
                    for a, b in ((i + k, i), (i - k, i), (i, i + k), (i, i - k)):
                        if 0 <= a < n and 0 <= b < n and lv[a, b] < 2:
                            lv[a, b] = rng.integers(2, 4)
        else:
            raise ValueError(f"stepfield: unknown layout {layout!r}, expected one of {LAYOUTS}")
        # enforce the adjacency rule by raising low neighbours of tall posts
        for _ in range(4):
            changed = False
            for i in range(n):
                for j in range(n):
                    for a, b in ((i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)):
                        if 0 <= a < n and 0 <= b < n and lv[a, b] - lv[i, j] > MAX_STEP:
                            lv[i, j] = lv[a, b] - MAX_STEP
                            changed = True
            if not changed:
                break
        return lv

    def _cells(self, tile: TileSpec, diff: DifficultyParams):
        post = diff.extra.get("post_size", POST_SIZE)
        lv = self._levels(diff)
        n = lv.shape[0]
        unit = diff.height / (N_LEVELS - 1)
        x0 = -n * post / 2
        y0 = -n * post / 2
        for i in range(n):
            for j in range(n):
                if lv[i, j] > 0:
                    yield x0 + post * (i + 0.5), y0 + post * (j + 0.5), post, tile.base_z + unit * lv[i, j]

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        for k, (cx, cy, post, top) in enumerate(self._cells(tile, diff)):
            add_ground_slab(stage, f"{prim_path}/post_{k}", cx, cy, post, post, top, tile)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for cx, cy, post, top in self._cells(tile, diff):
            paint_rect(h, tile, cx, cy, post, post, top)
        return h

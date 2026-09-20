"""Common interface every obstacle-type class implements.

Heightmap axis convention (matches `MapHelper` in FTR-Benchmark's terrain.py,
which indexes `self.map[x_index, y_index]`): a tile's local heightmap array has
shape `(tile.nx, tile.ny)` where axis 0 walks world X (the course/travel
direction, `tile.width` long) and axis 1 walks world Y (the lane direction,
`tile.depth` long).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, ClassVar

import numpy as np


@dataclass(frozen=True)
class TileSpec:
    """Where/how big a tile is, in the tile's own local frame.

    Local frame: origin (0, 0) is the tile's horizontal center; z=0 is the
    world Z of the shared base slab's top surface (`base_z`), i.e. obstacle
    geometry/heightmap values are authored relative to `base_z`, not to
    world 0.
    """

    width: float  # X extent, travel/course direction (e.g. 5.0)
    depth: float  # Y extent, lane direction (e.g. 3.333...)
    cell_size: float
    base_z: float  # world Z of the flat base slab's top surface

    @property
    def nx(self) -> int:
        return round(self.width / self.cell_size)

    @property
    def ny(self) -> int:
        return round(self.depth / self.cell_size)


@dataclass(frozen=True)
class DifficultyParams:
    """What 'difficulty' means for one tile instance (one repeat of a row)."""

    t: float  # normalized progress across the row's repeats, in [0, 1], 0=easiest
    height: float  # lerp(min_height, max_height, t) — primary obstacle amplitude (m)
    seed: int  # deterministic per-tile seed, derived from (course_seed, row, col)
    extra: dict[str, Any] = field(default_factory=dict)  # type-specific knobs
    col_index: int = 0  # this tile's repeat index within its row — for obstacles that need a
    # deterministic pattern ACROSS repeats (e.g. alternating left/right), not just per-tile
    # randomness (which diff.rng() already covers)

    def rng(self) -> np.random.Generator:
        return np.random.default_rng(self.seed)

    def lerp_extra(self, key_min: str, key_max: str, default: float = 0.0) -> float:
        """Interpolate an `extra` min/max pair (e.g. min_density/max_density) at `t`."""
        lo = self.extra.get(key_min, default)
        hi = self.extra.get(key_max, default)
        return lo + (hi - lo) * self.t

    @property
    def mirror_sign(self) -> float:
        """-1 on odd repeats when the row sets `extra.mirror_alternate: true`,
        else +1. Asymmetric obstacles (anything with a crossing angle or a
        raised side) multiply their angle/side by this so the course shows
        every asymmetric feature in BOTH handednesses at the same difficulty
        — a policy that only ever meets one sign learns a handed strategy
        (custom_mixed's diagonal_trunk scored 0.97 at -45 deg and 0.31 at
        +45 deg, mirror images of the same geometry)."""
        if self.extra.get("mirror_alternate", False) and self.col_index % 2 == 1:
            return -1.0
        return 1.0


class Obstacle(ABC):
    """One instance = one obstacle TYPE. No state kept between tiles — all
    per-tile variation flows through `DifficultyParams`, so generation stays
    pure-functional and reproducible from `terrain_config.yaml` + its seed.
    """

    name: ClassVar[str]

    @abstractmethod
    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        """Add prim(s) under `prim_path` (an Xform already positioned at this
        tile's location within the course). Author geometry as if the tile
        were centered at local (0, 0, tile.base_z). Must call
        `UsdPhysics.CollisionAPI.Apply()` on every collidable prim created;
        must NOT set any translate on `prim_path` itself (the assembler owns
        that) — only local offsets on children.
        """

    @abstractmethod
    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        """Return a float32 array of shape (tile.nx, tile.ny) giving absolute
        world-Z ground height at each cell (i.e. `tile.base_z` plus whatever
        local bump/dip this obstacle adds), using the same `diff.height` /
        `diff.extra` the USD used, so mesh and heightmap agree.
        """

    def num_repeats_hint(self, requested: int) -> int:
        """Override only if a type has a natural minimum repeat count."""
        return requested

    def birth_offsets(self, tile: TileSpec, diff: DifficultyParams) -> tuple[float, float]:
        """Extra Z offset (meters) added on top of the birth clearance Z for
        this tile's (start, target) points. Default (0, 0) — flat spawn/goal
        margins at ground level. Override when start or target instead sits
        on a raised feature (e.g. a stair platform).
        """
        return 0.0, 0.0

    def build_lips(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams, spec) -> int | None:
        """Obstacle-specific high-friction "lips" (see `edges.py`) under
        `prim_path`; return their count. The default `None` tells the assembler
        to derive lips from the heightmap's rising edges instead, which is
        right for anything built from boxes and wrong for a smooth cylinder.
        """
        return None

    def flat_heightmap(self, tile: TileSpec) -> np.ndarray:
        """Convenience: an all-`base_z` patch, for obstacles/subclasses to start from."""
        return np.full((tile.nx, tile.ny), tile.base_z, dtype=np.float32)


class MeshObstacle(Obstacle):
    """Base for obstacles whose surface is defined analytically per cell
    (slopes, cross-slopes, rotated staircases, waves) rather than as a set of
    axis-aligned boxes. Subclasses implement only `build_heightmap`; the USD
    is a triangle mesh of exactly that array (see
    `usd_utils.add_heightfield_mesh`) over a flat ground slab at the patch's
    minimum height, so physics and observation agree by construction.
    `extra.mesh_stride` (default 1 = every 5 cm cell) coarsens the mesh for
    smooth surfaces where the collision triangles need not follow every
    cell; keep 1 for anything with steps or edges.
    """

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        from ftr_terrain_gen.usd_utils import add_ground_slab, add_heightfield_mesh

        h = self.build_heightmap(tile, diff)
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, float(h.min()), tile)
        add_heightfield_mesh(stage, f"{prim_path}/surface", h, tile, stride=int(diff.extra.get("mesh_stride", 1)))

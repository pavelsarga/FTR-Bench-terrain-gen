"""Per-row obstacle variants, applied in `offset.resolve_obstacle` before the feature offset.

`extra.lowered: true` — the obstacle MIRRORED about the ground plane: h' = 2 base_z - h. A raised
platform becomes a pit, cobblestones become a sunken rough field, a stump field a set of dimples,
a wave starts with a trough. It turns an up-first obstacle into a down-first one with the same
shape statistics, which the v2 courses lack: on mixed_v2_straight the first sharp edge met goes
up on 73 % of drives, and every drop there comes right after a climb of the same height
(misc/metric_research README §5e). The USD is a triangle mesh of the mirrored heightmap over a
slab at its minimum, exactly as `MeshObstacle` builds its surfaces, so physics and observation
agree by construction for every base type; step walls become one-cell (5 cm) bevels. Edge lips
are derived from the mirrored heightmap (the obstacle's own `build_lips` would describe the
unmirrored geometry), and start/target Z offsets are negated.

`extra.sample: {key: [lo, hi], ...}` — draws each listed `extra` key per tile, uniformly in
[lo, hi], from a seeded stream separate from the obstacle's own `diff.rng()`, and hands the base
obstacle a copy of `diff` with those values. Lengths (every key not ending in `_ratio`) are
snapped to the 5 cm cell. Used to vary platform / pit / plateau lengths so the drop does not
follow the rise at a fixed distance, and `raised_platform.exit_height_ratio` so it is not the
same height either. Rows without `sample` are untouched (no draw from any stream).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec

_SAMPLE_SALT = 777001  # separate from offset.py's _SEED_SALT and the obstacle's own rng
CELL = 0.05


class SampledObstacle(Obstacle):
    def __init__(self, base: Obstacle):
        self.base = base
        self.name = base.name

    @staticmethod
    def sampled_diff(diff: DifficultyParams) -> DifficultyParams:
        spec = diff.extra.get("sample") or {}
        rng = np.random.default_rng((diff.seed + _SAMPLE_SALT) % (2**32))
        extra = dict(diff.extra)
        for key in sorted(spec):  # sorted: the draw order must not depend on YAML key order
            lo, hi = (float(v) for v in spec[key])
            v = float(rng.uniform(lo, hi))
            extra[key] = v if key.endswith("_ratio") else round(v / CELL) * CELL
        return replace(diff, extra=extra)

    def build_usd(self, stage, prim_path, tile, diff):
        self.base.build_usd(stage, prim_path, tile, self.sampled_diff(diff))

    def build_heightmap(self, tile, diff):
        return self.base.build_heightmap(tile, self.sampled_diff(diff))

    def build_lips(self, stage, prim_path, tile, diff, spec):
        return self.base.build_lips(stage, prim_path, tile, self.sampled_diff(diff), spec)

    def birth_offsets(self, tile, diff):
        return self.base.birth_offsets(tile, self.sampled_diff(diff))

    def num_repeats_hint(self, requested):
        return self.base.num_repeats_hint(requested)


class LoweredObstacle(Obstacle):
    def __init__(self, base: Obstacle):
        self.base = base
        self.name = base.name

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        return (2.0 * tile.base_z - self.base.build_heightmap(tile, diff)).astype(np.float32)

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        from ftr_terrain_gen.usd_utils import add_ground_slab, add_heightfield_mesh

        h = self.build_heightmap(tile, diff)
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, float(h.min()), tile)
        add_heightfield_mesh(stage, f"{prim_path}/surface", h, tile, stride=int(diff.extra.get("mesh_stride", 1)))

    def build_lips(self, stage, prim_path, tile, diff, spec):
        return None  # derive from the mirrored heightmap

    def birth_offsets(self, tile, diff):
        a, b = self.base.birth_offsets(tile, diff)
        return -a, -b

    def num_repeats_hint(self, requested):
        return self.base.num_repeats_hint(requested)


def apply_variants(base: Obstacle, diff: DifficultyParams) -> Obstacle:
    if diff.extra.get("sample"):
        base = SampledObstacle(base)
    if diff.extra.get("lowered"):
        base = LoweredObstacle(base)
    return base

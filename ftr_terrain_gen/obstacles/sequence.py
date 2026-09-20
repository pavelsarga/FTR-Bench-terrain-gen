from __future__ import annotations

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec


class Sequence(Obstacle):
    """Several obstacles chained along X inside ONE tile, in the order the
    robot meets them (the first listed part sits at the +X / spawn end).
    `extra.parts` is a list of row-like dicts — `{type, min_height,
    max_height, extra, width}` — each generated as if in its own narrower
    tile of `width` (default: an equal share of `tile.width`), then placed
    side by side. Every part is graded by the same `diff.t` as the row, and
    gets its own deterministic seed. Obstacle-specific `birth_offsets`
    come from the first part (start) and last part (target).

    Why: the course grid is uniform, so a 1.0 m log in an 8.8 m tile is
    7.8 m of flat runway — which is how a policy learns to drive flat out.
    Pairing two short features with ~1 m between them makes the approach
    speed to the second one a decision, and gives the holdout course new
    COMBINATIONS of training obstacles rather than new obstacles.
    """

    name = "sequence"

    def _parts(self, tile: TileSpec, diff: DifficultyParams):
        from ftr_terrain_gen.registry import get_obstacle

        parts = diff.extra.get("parts")
        if not parts:
            raise ValueError("sequence: extra.parts must list at least one obstacle")
        widths = [p.get("width") for p in parts]
        n_free = sum(w is None for w in widths)
        used = sum(w for w in widths if w is not None)
        if n_free:
            share = (tile.width - used) / n_free
            widths = [share if w is None else w for w in widths]
        if abs(sum(widths) - tile.width) > 1e-6:
            raise ValueError(f"sequence: part widths {widths} must sum to tile.width={tile.width}")
        x = tile.width / 2  # first part at the +X (spawn) end
        for k, (part, w) in enumerate(zip(parts, widths)):
            x_center = x - w / 2
            x -= w
            sub_tile = TileSpec(width=w, depth=tile.depth, cell_size=tile.cell_size, base_z=tile.base_z)
            lo, hi = part.get("min_height", 0.0), part.get("max_height", 0.0)
            sub_extra = dict(part.get("extra", {}))
            if "mirror_alternate" in diff.extra and "mirror_alternate" not in sub_extra:
                sub_extra["mirror_alternate"] = diff.extra["mirror_alternate"]
            sub_diff = DifficultyParams(
                t=diff.t, height=lo + (hi - lo) * diff.t, seed=(diff.seed + 7919 * (k + 1)) % (2**32),
                extra=sub_extra, col_index=diff.col_index,
            )
            yield get_obstacle(part["type"]), sub_tile, sub_diff, x_center

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        from pxr import Gf, UsdGeom

        for k, (obstacle, sub_tile, sub_diff, x_center) in enumerate(self._parts(tile, diff)):
            part_path = f"{prim_path}/part_{k}_{obstacle.name}"
            xform = UsdGeom.Xform.Define(stage, part_path)
            UsdGeom.Xformable(xform).AddTranslateOp().Set(Gf.Vec3d(x_center, 0.0, 0.0))
            obstacle.build_usd(stage, part_path, sub_tile, sub_diff)

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        for obstacle, sub_tile, sub_diff, x_center in self._parts(tile, diff):
            patch = obstacle.build_heightmap(sub_tile, sub_diff)
            i0 = int(round((x_center - sub_tile.width / 2 + tile.width / 2) / tile.cell_size))
            i1 = min(i0 + patch.shape[0], tile.nx)
            h[i0:i1, :] = patch[: i1 - i0, :]
        return h

    def birth_offsets(self, tile: TileSpec, diff: DifficultyParams) -> tuple[float, float]:
        parts = list(self._parts(tile, diff))
        start = parts[0][0].birth_offsets(parts[0][1], parts[0][2])[0]
        target = parts[-1][0].birth_offsets(parts[-1][1], parts[-1][2])[1]
        return start, target

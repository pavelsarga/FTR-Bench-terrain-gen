"""Visual markings for a course, written to a SEPARATE `usd/<name>_decor.usd`
that FTR-Benchmark's `Terrain` only loads when the env asks for it
(`terrain_decor: true` — the eval wrappers set it for GUI runs), so the
headless training sim never pays for it. Everything here is a decal: no
collision, a few millimetres above the surface, following the composited
heightmap so it also reads correctly on the sloped rows.

Per-course `decor:` block in terrain_config.yaml (all optional, defaults shown):

    decor:
      tile_lines: true        # red lines on every tile boundary
      spawn_marks: true       # green disc at each start, blue square at each target
      path_lines: true        # red start->start lines over the obstacle
      hazard_frames: true     # black/yellow striped frame around each goal zone
      line_width: 0.08
      lift: 0.004             # metres above the surface
      colors: {tile_line: [0.85, 0.1, 0.1], spawn: [0.1, 0.8, 0.2], goal: [0.2, 0.4, 1.0],
               path: [0.85, 0.1, 0.1], hazard_a: [0.05, 0.05, 0.05], hazard_b: [1, 0.85, 0]}

`decor: false` disables the file altogether.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from ftr_terrain_gen.grid import CourseGrid

DEFAULTS = {
    "tile_lines": True,
    "spawn_marks": True,
    "path_lines": True,
    "hazard_frames": True,
    "line_width": 0.08,
    "lift": 0.004,
    "goal_radius": 0.4,  # CrossingEnv's success radius — the hazard frame is drawn around it
    "colors": {
        "tile_line": (0.85, 0.1, 0.1),
        "spawn": (0.1, 0.8, 0.2),
        "goal": (0.2, 0.4, 1.0),
        "path": (0.85, 0.1, 0.1),
        "hazard_a": (0.05, 0.05, 0.05),
        "hazard_b": (1.0, 0.85, 0.0),
    },
}


def resolve_decor_cfg(raw) -> dict | None:
    if raw is False:
        return None
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if isinstance(raw, dict):
        for k, v in raw.items():
            if k == "colors" and isinstance(v, dict):
                cfg["colors"].update({kk: tuple(vv) for kk, vv in v.items()})
            else:
                cfg[k] = v
    return cfg


class _Sampler:
    """Height of the composited heightmap at a world (x, y), for lifting decals onto the surface."""

    def __init__(self, grid: CourseGrid, heightmap: np.ndarray):
        self.grid, self.h = grid, heightmap

    def __call__(self, x: float, y: float) -> float:
        i, j = self.grid.world_to_index((x, y))
        i = min(max(i, 0), self.h.shape[0] - 1)
        j = min(max(j, 0), self.h.shape[1] - 1)
        z = float(self.h[i, j])
        return z if z > self.grid.base_z - 1.5 else self.grid.base_z  # sentinel -> ground


def _polyline(sampler: _Sampler, p0, p1, step: float, lift: float):
    n = max(2, int(math.ceil(math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / step)) + 1)
    xs = np.linspace(p0[0], p1[0], n)
    ys = np.linspace(p0[1], p1[1], n)
    return [(float(x), float(y), sampler(x, y) + lift) for x, y in zip(xs, ys)]


def build_decor_usd(grid: CourseGrid, heightmap: np.ndarray, birth: list[dict], cfg: dict, usd_path: Path) -> int:
    """Write the decal stage. Returns the number of prims authored."""
    from pxr import Usd, UsdGeom

    from ftr_terrain_gen.usd_utils import add_cube, add_cylinder, add_ribbon

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    sample = _Sampler(grid, heightmap)
    colors = cfg["colors"]
    lw, lift = cfg["line_width"], cfg["lift"]
    step = grid.cell_size
    n_prims = 0
    hx, hy = grid.course_width_x / 2, grid.course_depth_y / 2

    if cfg["tile_lines"]:
        k = 0
        for c in range(grid.repeats + 1):
            x = -hx + c * grid.tile_width
            add_ribbon(stage, f"/World/tile_lines/x_{k:03d}", _polyline(sample, (x, -hy), (x, hy), step, lift), lw, colors["tile_line"])
            k += 1
        for r in range(grid.n_rows + 1):
            y = -hy + r * grid.tile_depth
            add_ribbon(stage, f"/World/tile_lines/y_{k:03d}", _polyline(sample, (-hx, y), (hx, y), step, lift), lw, colors["tile_line"])
            k += 1
        n_prims += k

    if cfg["path_lines"]:
        # one line per start point, to the OPPOSING start on the other side of the tile with
        # the same lateral offset (three parallel lines when spawn_lateral_jitter is on; for
        # a goal-offset row the opposing starts are shifted, so the lines are diagonal but
        # still parallel to each other). Only forward entries are used — the reverse leg
        # is the same line drawn backwards.
        for e in birth:
            if e["start_orient"][2] == 0.0:
                continue
            s_pt, t_pt = e["start_point"], e["target_point"]
            row = int(round(s_pt[1] / grid.tile_depth + (grid.n_rows - 1) / 2))
            lane_y = (row - (grid.n_rows - 1) / 2) * grid.tile_depth
            jitter = s_pt[1] - lane_y
            opposing = (t_pt[0], t_pt[1] + jitter)  # where the reverse leg with this jitter starts
            add_ribbon(stage, f"/World/path_lines/p_{n_prims:04d}", _polyline(sample, s_pt, opposing, step, 2 * lift), lw * 0.75, colors["path"])
            n_prims += 1

    if cfg["spawn_marks"]:
        for k, e in enumerate(birth):
            s, t = e["start_point"], e["target_point"]
            add_cylinder(stage, f"/World/spawn_marks/start_{k:04d}", 0.3, 0.002, "Z",
                         (s[0], s[1], sample(s[0], s[1]) + 3 * lift), collide=False, color=colors["spawn"])
            add_cube(stage, f"/World/spawn_marks/goal_{k:04d}", (0.36, 0.36, 0.002),
                     (t[0], t[1], sample(t[0], t[1]) + 4 * lift), collide=False, color=colors["goal"])
            n_prims += 2

    if cfg["hazard_frames"]:
        # one closed ribbon per goal zone, striped black/yellow per 0.15 m segment
        half = cfg["goal_radius"] + 0.1
        seg = 0.15
        seen = set()
        for e in birth:
            t = e["target_point"]
            key = (round(t[0], 2), round(t[1], 2))
            if key in seen:
                continue
            seen.add(key)
            z = sample(t[0], t[1]) + 3 * lift
            corners = [(t[0] - half, t[1] - half), (t[0] + half, t[1] - half),
                       (t[0] + half, t[1] + half), (t[0] - half, t[1] + half), (t[0] - half, t[1] - half)]
            pts, seg_colors = [], []
            n_seg = max(2, int(round(2 * half / seg)))
            for c0, c1 in zip(corners[:-1], corners[1:]):
                for m in range(n_seg):
                    a = m / n_seg
                    pts.append((c0[0] + (c1[0] - c0[0]) * a, c0[1] + (c1[1] - c0[1]) * a, z))
                    seg_colors.append(colors["hazard_a"] if m % 2 == 0 else colors["hazard_b"])
            pts.append((corners[-1][0], corners[-1][1], z))
            add_ribbon(stage, f"/World/hazard/f_{n_prims:05d}", pts, 0.06, colors["hazard_a"], segment_colors=seg_colors)
            n_prims += 1

    stage.Save()
    return n_prims

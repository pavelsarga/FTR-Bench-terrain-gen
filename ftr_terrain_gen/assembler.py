"""Builds the combined USD stage, composites the .map array, and writes the
output config.yaml / birth.json — the four files `Terrain` (in FTR-Benchmark's
terrain.py) expects for a terrain named `<name>`. Also copies the source
terrain_config.yaml verbatim into `gen_config/<name>.yaml` and saves an SVG
heightmap plot (tile grid + start/goal markers) to `plot/<name>.svg` —
neither is read by `Terrain`/FTR-Bench, both are logging/eval-time artifacts
(config provenance, quick visual reference for notebooks).

Optional top-level `tile_dividers: {height, width}` in terrain_config.yaml adds
a wall at every internal tile boundary within each row (see
`compute_tile_dividers`) — off by default, applies to every row. A row's own
`extra: {side_walls: {height, width}}` adds a wall along just THAT row's two Y
edges, per tile (see `compute_side_walls`) — opt-in per obstacle/row, not
course-wide.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np
import yaml

from ftr_terrain_gen.birth import build_birth
from ftr_terrain_gen.grid import CourseGrid
from ftr_terrain_gen.registry import get_obstacle


def _sanitize(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]", "_", name)


def build_map(grid: CourseGrid) -> np.ndarray:
    shape = grid.map_shape
    arr = np.full(shape, -1.0, dtype=np.float32)
    for tile in grid.iter_tiles():
        obstacle = get_obstacle(tile.obstacle_type)
        tile_spec = grid.tile_spec()
        patch = obstacle.build_heightmap(tile_spec, tile.diff)
        i0, j0 = grid.world_to_index((tile.x_center - grid.tile_width / 2, tile.y_center - grid.tile_depth / 2))
        i1, j1 = i0 + patch.shape[0], j0 + patch.shape[1]
        ci0, cj0 = max(i0, 0), max(j0, 0)
        ci1, cj1 = min(i1, shape[0]), min(j1, shape[1])
        if ci1 <= ci0 or cj1 <= cj0:
            continue
        arr[ci0:ci1, cj0:cj1] = patch[ci0 - i0 : ci1 - i0, cj0 - j0 : cj1 - j0]
    return arr


def compute_tile_dividers(
    grid: CourseGrid, heightmap: np.ndarray, divider_cfg: dict
) -> list[tuple[float, float, float, float, float]]:
    """One wall per INTERNAL column boundary within each row (between
    adjacent obstacle "spots" along a lane), so a robot can't drive/slide
    from one tile's obstacle straight into the next — stairs_ascent_descent
    in particular has its ground-level or platform-level ends flush with
    the neighboring tile's, with nothing physically stopping that. Returns
    (x_center, y_center, width, depth, top_z) world-space specs; height is
    `divider_cfg['height']` above the TALLER of the two adjacent tiles'
    surface at that exact boundary, sampled from the already-composited
    `heightmap` (not assumed), so it clears whichever side is higher.
    """
    height = divider_cfg.get("height", 1.0)
    width = divider_cfg.get("width", 0.1)
    sample_offset = width / 2 + 3 * grid.cell_size  # clear of the divider's own footprint
    dividers = []
    for row_index in range(grid.n_rows):
        y_center = (row_index - (grid.n_rows - 1) / 2) * grid.tile_depth
        for col in range(grid.repeats - 1):
            boundary_x = grid._grid_x_center(col) + grid.tile_width / 2
            i_left, j = grid.world_to_index((boundary_x - sample_offset, y_center))
            i_right, _ = grid.world_to_index((boundary_x + sample_offset, y_center))
            local_top = float(max(heightmap[i_left, j], heightmap[i_right, j]))
            dividers.append((boundary_x, y_center, width, grid.tile_depth, local_top + height))
    return dividers


def paint_wall_segments(
    arr: np.ndarray, grid: CourseGrid, segments: list[tuple[float, float, float, float, float]]
) -> None:
    """Shared by tile dividers and side walls — both are just a list of
    (x_center, y_center, width, depth, top_z) world-space rects to stamp
    into the full-course heightmap array."""
    for x_center, y_center, width, depth, top_z in segments:
        i0, j0 = grid.world_to_index((x_center - width / 2, y_center - depth / 2))
        i1, j1 = grid.world_to_index((x_center + width / 2, y_center + depth / 2))
        ci0, cj0 = max(i0, 0), max(j0, 0)
        ci1, cj1 = min(i1, arr.shape[0]), min(j1, arr.shape[1])
        if ci1 > ci0 and cj1 > cj0:
            arr[ci0:ci1, cj0:cj1] = top_z


def compute_side_walls(grid: CourseGrid, heightmap: np.ndarray) -> list[tuple[float, float, float, float, float]]:
    """One wall segment per tile, along EACH of a row's two Y edges (the
    boundary between that lane and whatever's next to it — another row or
    the course border) — stops a robot sliding sideways off a lane instead
    of just driving straight through the obstacles. Opt-in PER ROW/OBSTACLE
    via that row's own `extra: {side_walls: {height, width}}` — most rows
    have no such key and get no walls; only checks `grid.rows`, not a
    course-wide switch. One segment per tile (not one long slab per row) so
    height can follow that tile's own local terrain if it's graded. Returns
    (x_center, y_center, width, depth, top_z) world-space specs; height is
    `side_walls['height']` above the tallest point anywhere in that tile
    (sampled from the already-composited `heightmap`), depth is
    `side_walls['width']` (wall thickness along Y).
    """
    walls = []
    for row_index, row in enumerate(grid.rows):
        cfg = row.extra.get("side_walls")
        if not cfg:
            continue
        height = cfg.get("height", 1.0)
        width = cfg.get("width", 0.1)
        y_center = (row_index - (grid.n_rows - 1) / 2) * grid.tile_depth
        y_edges = (y_center - grid.tile_depth / 2, y_center + grid.tile_depth / 2)
        for col in range(grid.repeats):
            x_center = grid._grid_x_center(col)
            i0, j0 = grid.world_to_index((x_center - grid.tile_width / 2, y_center - grid.tile_depth / 2))
            i1, j1 = grid.world_to_index((x_center + grid.tile_width / 2, y_center + grid.tile_depth / 2))
            ci0, cj0 = max(i0, 0), max(j0, 0)
            ci1, cj1 = min(i1, heightmap.shape[0]), min(j1, heightmap.shape[1])
            local_top = float(heightmap[ci0:ci1, cj0:cj1].max()) if ci1 > ci0 and cj1 > cj0 else grid.base_z
            for y_edge in y_edges:
                walls.append((x_center, y_edge, grid.tile_width, width, local_top + height))
    return walls


def build_usd(
    grid: CourseGrid,
    usd_path: Path,
    dividers: list[tuple[float, float, float, float, float]] = (),
    side_walls: list[tuple[float, float, float, float, float]] = (),
) -> None:
    from pxr import Gf, Usd, UsdGeom

    from ftr_terrain_gen.usd_utils import add_ground_slab

    stage = Usd.Stage.CreateNew(str(usd_path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    root = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(root.GetPrim())

    tile_spec = grid.tile_spec()
    for tile in grid.iter_tiles():
        obstacle = get_obstacle(tile.obstacle_type)
        row_path = f"/World/row_{tile.row_index:02d}_{_sanitize(tile.obstacle_type)}"
        tile_path = f"{row_path}/tile_{tile.col_index:02d}"
        tile_xform = UsdGeom.Xform.Define(stage, tile_path)
        UsdGeom.Xformable(tile_xform).AddTranslateOp().Set(Gf.Vec3d(tile.x_center, tile.y_center, 0.0))
        obstacle.build_usd(stage, tile_path, tile_spec, tile.diff)

    for i, (x_center, y_center, width, depth, top_z) in enumerate(dividers):
        add_ground_slab(stage, f"/World/tile_dividers/div_{i:03d}", x_center, y_center, width, depth, top_z, tile_spec)

    for i, (x_center, y_center, width, depth, top_z) in enumerate(side_walls):
        add_ground_slab(stage, f"/World/side_walls/wall_{i:03d}", x_center, y_center, width, depth, top_z, tile_spec)

    stage.Save()


def build_plot(grid: CourseGrid, heightmap: np.ndarray, birth: list[dict], plot_path: Path) -> None:
    """Save a colored heightmap image — no title/axis ticks/labels — with a
    grid overlay at each obstacle tile's boundaries and start (circle) /
    goal (star) markers from birth.json, for quick visual reference in
    notebooks. Crops off the `-1.0` sentinel border ring first: left in,
    its huge drop below the real terrain heights would stretch the colormap
    so far that the actual steps/height differences we care about all wash
    out to nearly the same shade.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    b = round(grid.border_width / grid.cell_size)
    interior = heightmap[b:-b, b:-b] if b > 0 else heightmap

    half_x = grid.course_width_x / 2
    half_y = grid.course_depth_y / 2
    aspect = grid.course_depth_y / grid.course_width_x
    fig_w = 12.0
    fig = Figure(figsize=(fig_w, max(fig_w * aspect, 1.0)), dpi=150)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(
        interior.T, origin="lower", cmap="terrain", interpolation="nearest",
        extent=(-half_x, half_x, -half_y, half_y),
    )

    for k in range(grid.repeats + 1):
        ax.axvline(-half_x + k * grid.tile_width, color="black", linewidth=1.5)
    for r in range(grid.n_rows + 1):
        ax.axhline(-half_y + r * grid.tile_depth, color="black", linewidth=1.5)

    starts = np.array([e["start_point"][:2] for e in birth])
    targets = np.array([e["target_point"][:2] for e in birth])
    ax.scatter(starts[:, 0], starts[:, 1], marker="o", color="lime", edgecolors="black", s=40, zorder=5)
    ax.scatter(targets[:, 0], targets[:, 1], marker="*", color="red", edgecolors="black", s=90, zorder=5)

    ax.set_xlim(-half_x, half_x)
    ax.set_ylim(-half_y, half_y)
    ax.axis("off")
    fig.savefig(plot_path, format="svg")


def build_config_yaml(grid: CourseGrid, camera: dict) -> dict:
    return {
        "prim_config": {
            "set_attrs": [
                {"prim_path": "terrain", "attr_name": "xformOp:translate", "value": [0.0, 0.0, 0.0]},
                {"prim_path": "terrain", "attr_name": "xformOp:orient", "value": [0, 0, 0, 1.0]},
            ]
        },
        "map": {
            "lower": list(grid.map_lower),
            "upper": list(grid.map_upper),
            "cell_size": grid.cell_size,
        },
        "camera": camera,
    }


def generate(
    course_cfg: dict,
    output_dir: Path,
    source_config_path: Path | None = None,
    overwrite: bool = False,
    dry_run: bool = False,
) -> None:
    name = course_cfg["name"]
    tile_cfg = course_cfg.get("tile", {})
    from ftr_terrain_gen.grid import RowConfig

    rows = [
        RowConfig(
            type=r["type"],
            min_height=r.get("min_height", 0.0),
            max_height=r.get("max_height", 0.0),
            extra=r.get("extra", {}),
        )
        for r in course_cfg["rows"]
    ]
    grid = CourseGrid(
        rows=rows,
        repeats=course_cfg["repeats"],
        tile_width=tile_cfg.get("width", 5.0),
        tile_depth=tile_cfg.get("depth", 10.0 / 3.0),
        cell_size=course_cfg.get("cell_size", 0.05),
        base_z=course_cfg.get("base_z", 0.5),
        border_width=course_cfg.get("border_width", 2.0),
        course_seed=course_cfg.get("seed", 0),
    )

    print(f"[generate_terrain] course '{name}': {grid.n_rows} rows x {grid.repeats} repeats")
    print(f"[generate_terrain] extent: {grid.course_width_x:.2f} x {grid.course_depth_y:.2f} m "
          f"(map {grid.map_shape}, lower={grid.map_lower}, upper={grid.map_upper})")
    for row in rows:
        print(f"  row: {row.type:24s} height=[{row.min_height},{row.max_height}]")

    heightmap = build_map(grid)
    divider_cfg = course_cfg.get("tile_dividers")
    dividers = compute_tile_dividers(grid, heightmap, divider_cfg) if divider_cfg else []
    if dividers:
        paint_wall_segments(heightmap, grid, dividers)
    side_walls = compute_side_walls(grid, heightmap)
    if side_walls:
        paint_wall_segments(heightmap, grid, side_walls)
    birth = build_birth(
        grid,
        z_clearance=course_cfg.get("birth_z_clearance", 0.125),
        birth_clearance=course_cfg.get("birth_clearance", 1.0),
    )
    config_yaml = build_config_yaml(grid, course_cfg.get("camera", {"position": [-40, 40, 25], "target": [0, 5, -5]}))

    print(f"[generate_terrain] heightmap stats: min={heightmap.min():.3f} max={heightmap.max():.3f}")
    print(f"[generate_terrain] birth entries: {len(birth)}")
    if dividers:
        print(f"[generate_terrain] tile dividers: {len(dividers)}")
    if side_walls:
        print(f"[generate_terrain] side walls: {len(side_walls)}")

    if dry_run:
        print("[generate_terrain] --dry-run: not writing any files")
        return

    for sub in ("config", "usd", "map", "birth", "gen_config", "plot"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    usd_path = output_dir / "usd" / f"{name}.usd"
    map_path = output_dir / "map" / f"{name}.map"
    config_path = output_dir / "config" / f"{name}.yaml"
    birth_path = output_dir / "birth" / f"{name}.json"
    gen_config_path = output_dir / "gen_config" / f"{name}.yaml"
    plot_path = output_dir / "plot" / f"{name}.svg"

    for p in (usd_path, map_path, config_path, birth_path, gen_config_path, plot_path):
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} already exists — pass --overwrite to replace it")

    build_usd(grid, usd_path, dividers, side_walls)
    with open(map_path, "wb") as f:
        np.save(f, heightmap)
    with open(config_path, "w") as f:
        yaml.safe_dump(config_yaml, f, sort_keys=False)
    with open(birth_path, "w") as f:
        json.dump(birth, f, indent=2)
    if source_config_path is not None:
        shutil.copyfile(source_config_path, gen_config_path)
    else:
        with open(gen_config_path, "w") as f:
            yaml.safe_dump(course_cfg, f, sort_keys=False)
    build_plot(grid, heightmap, birth, plot_path)

    print(f"[generate_terrain] wrote {usd_path}, {map_path}, {config_path}, {birth_path}, {gen_config_path}, {plot_path}")

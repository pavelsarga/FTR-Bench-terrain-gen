"""Builds the combined USD stage, composites the .map array, and writes the
output config.yaml / birth.json — the four files `Terrain` (in FTR-Benchmark's
terrain.py) expects for a terrain named `<name>`.
"""

from __future__ import annotations

import json
import re
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


def build_usd(grid: CourseGrid, usd_path: Path) -> None:
    from pxr import Gf, Usd, UsdGeom

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

    stage.Save()


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


def generate(course_cfg: dict, output_dir: Path, overwrite: bool = False, dry_run: bool = False) -> None:
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
    birth = build_birth(
        grid,
        z_clearance=course_cfg.get("birth_z_clearance", 0.125),
        birth_clearance=course_cfg.get("birth_clearance", 0.5),
    )
    config_yaml = build_config_yaml(grid, course_cfg.get("camera", {"position": [-40, 40, 25], "target": [0, 5, -5]}))

    print(f"[generate_terrain] heightmap stats: min={heightmap.min():.3f} max={heightmap.max():.3f}")
    print(f"[generate_terrain] birth entries: {len(birth)}")

    if dry_run:
        print("[generate_terrain] --dry-run: not writing any files")
        return

    for sub in ("config", "usd", "map", "birth"):
        (output_dir / sub).mkdir(parents=True, exist_ok=True)

    usd_path = output_dir / "usd" / f"{name}.usd"
    map_path = output_dir / "map" / f"{name}.map"
    config_path = output_dir / "config" / f"{name}.yaml"
    birth_path = output_dir / "birth" / f"{name}.json"

    for p in (usd_path, map_path, config_path, birth_path):
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} already exists — pass --overwrite to replace it")

    build_usd(grid, usd_path)
    with open(map_path, "wb") as f:
        np.save(f, heightmap)
    with open(config_path, "w") as f:
        yaml.safe_dump(config_yaml, f, sort_keys=False)
    with open(birth_path, "w") as f:
        json.dump(birth, f, indent=2)

    print(f"[generate_terrain] wrote {usd_path}, {map_path}, {config_path}, {birth_path}")

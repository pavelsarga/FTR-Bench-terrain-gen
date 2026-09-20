#!/usr/bin/env python3
"""CLI entrypoint: generate a procedural FTR-Benchmark terrain course.

Usage:
    python generate_terrain.py terrain_config.yaml --output-dir /path/to/ftr_envs/assets/terrain [--dry-run] [--overwrite]

--output-dir should point at the FTR-Benchmark terrain directory itself
(the one containing config/, usd/, map/, birth/ subfolders) so the emitted
files land where Terrain.__init__ (in terrain.py) expects them.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from ftr_terrain_gen.assembler import generate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("terrain_config", type=Path, help="Path to terrain_config.yaml")
    parser.add_argument("--output-dir", type=Path, required=True, help="ftr_envs/assets/terrain directory to write into")
    parser.add_argument("--dry-run", action="store_true", help="Run layout + heightmap generation only, print stats, write nothing (no pxr needed)")
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing <name>.* files")
    args = parser.parse_args()

    course_cfg, resolved = load_course_config(args.terrain_config)

    generate(
        course_cfg, args.output_dir,
        # a config that pulls rows from another file is copied to gen_config/ in its
        # RESOLVED form (env_type_registry reads the row list from that copy)
        source_config_path=None if resolved else args.terrain_config,
        overwrite=args.overwrite, dry_run=args.dry_run,
    )


def load_course_config(path: Path) -> tuple[dict, bool]:
    """Read a terrain config; `include_rows: <other config>` (relative to this
    file) prepends that course's rows to this one's, so a variant course
    (the turning-obstacle `*_full` course) shares its base rows with the
    straight-drive course by reference instead of by copy. Returns the
    config and whether anything was included."""
    with open(path) as f:
        course_cfg = yaml.safe_load(f)
    include = course_cfg.pop("include_rows", None)
    if not include:
        return course_cfg, False
    base_cfg, _ = load_course_config(path.parent / include)
    course_cfg["rows"] = list(base_cfg["rows"]) + list(course_cfg.get("rows", []))
    for key in ("tile", "cell_size", "base_z", "border_width", "repeats", "seed", "arena_band",
                "bidirectional", "spawn_lateral_jitter", "birth_clearance", "birth_z_clearance", "decor", "feature_offset"):
        course_cfg.setdefault(key, base_cfg[key]) if key in base_cfg else None
    return course_cfg, True


if __name__ == "__main__":
    main()

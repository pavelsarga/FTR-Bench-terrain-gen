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

    with open(args.terrain_config) as f:
        course_cfg = yaml.safe_load(f)

    generate(course_cfg, args.output_dir, overwrite=args.overwrite, dry_run=args.dry_run)


if __name__ == "__main__":
    main()

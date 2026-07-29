# FTR-Bench-terrain-generation

Procedural generator for FTR-Benchmark terrain courses. From a single
`terrain_config.yaml` (obstacle types, graded repeats, min/max height per
type) it produces the four files `Terrain` (in
`src/FTR-Benchmark/ftr_envs/assets/terrain/terrain.py`) expects:
`config/<name>.yaml`, `usd/<name>.usd`, `map/<name>.map`, `birth/<name>.json`.
It also copies the source `terrain_config.yaml` verbatim into
`gen_config/<name>.yaml` and saves an SVG heightmap plot (tile grid +
start/goal markers) to `plot/<name>.svg` — neither is read by
`Terrain`/FTR-Bench, both are logging/eval-time artifacts (config
provenance, quick visual reference for notebooks).

Standalone — generating files only needs `usd-core`/`numpy`, no Isaac Sim.

## Install

```bash
pip install -e .
# or, standalone:
pip install usd-core numpy pyyaml matplotlib
```

## Usage

```bash
python generate_terrain.py terrain_config_example.yaml \
    --output-dir /path/to/src/FTR-Benchmark/ftr_envs/assets/terrain \
    --dry-run          # layout + heightmap stats only, writes nothing

python generate_terrain.py terrain_config_example.yaml \
    --output-dir /path/to/src/FTR-Benchmark/ftr_envs/assets/terrain \
    --overwrite         # writes usd/map/config/birth for <name>
```

`--output-dir` should point at `ftr_envs/assets/terrain` so the emitted
subfolders line up with what `Terrain.__init__` expects. `--overwrite` is
required to replace existing files.

See `terrain_config_example.yaml` for the full schema.

## Obstacle catalog

- `raised_platform` — flat spawn, a raised platform, flat goal.
- `double_trench` — two trenches with level ground between them.
- `widening_trench` — one trench, width grades with difficulty (depth fixed).
- `log_crossing` — a raised block along the travel direction; the robot
  rides along its top rather than climbing over it.
- `twin_rails` — two thin rails across the lane with a gap between them,
  enclosed by guard walls.
- `diagonal_trunk` — a raised block crossing the lane at an angle (the
  angle itself grades with difficulty), enclosed by guard walls.
- `raised_stairs` — steps up to a platform, then back down.
- `lowered_stairs` — recessed mirror of `raised_stairs`: steps down to a
  platform, then back up.
- `cobblestones` — a square field split into a grid of blocks, each
  independently randomized in height.
- `rock_formation` — like `cobblestones` but spatially-correlated (noise
  based) heights with a ragged, non-rectangular footprint.
- `diagonal_pyramid` — a square field banded diagonally into rows that rise
  to a central peak and fall away again.
- `half_platform` — half of a square field (split along the lane width) is
  raised; which half alternates by repeat for asymmetric-loading training.
- `flat_patch` — the bare tile, no obstacle at all (rest/baseline lane).
- `stump` — a fixed-height smooth radial bump (derivative-of-sigmoid
  profile); its center slides sideways from slightly left of the lane's
  centerline to slightly right as difficulty grades.
- `stairs_ascent_descent` — a one-way staircase spanning the full tile,
  alternating ascending/descending by repeat (start below and goal on top,
  or vice versa) instead of mirroring up-then-down in one tile like
  `raised_stairs`.

Every obstacle (except `stairs_ascent_descent`, a one-way ramp by design) is
CENTERED in its tile — flat margins on both sides are equal, computed from
`tile.width` and the feature's own size, not an independent `front_margin`
knob. `birth.py`'s `birth_clearance` (default 1.0m — the spawn/goal point
sits exactly 1m in from the tile edge) then needs each margin to be big
enough on its own: MARV is 1.1m long, so its body also needs at least 0.6m
of clearance beyond its own half-length from that point to the nearest
riser or edge (>=1.15m of flat ground past the spawn point, in whichever
direction the obstacle is) — make the tile ("obstacle spot") wider if a
feature is large relative to `tile.width`, rather than shrinking margins.

Every obstacle exposes tunable defaults (widths, grid size, etc.) via a
row's `extra:` dict in `terrain_config.yaml` — see
`terrain_config_example.yaml` for worked examples, or each obstacle's file
under `ftr_terrain_gen/obstacles/` for its full parameter list.

New types implement the `Obstacle` interface in
`ftr_terrain_gen/obstacle_base.py` and register in `ftr_terrain_gen/registry.py`.

## Tests

```bash
pytest tests/
```

Covers grid layout math, heightmap sanity, and a `usd-core` smoke test
(collision APIs, bounding boxes). Physics/visual verification still needs
Isaac Sim.

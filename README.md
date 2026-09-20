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
- `lowered_platform` — the recessed mirror of `raised_platform`: a pit the
  robot drops into and has to climb back out of.

Sloped / analytic surfaces (built as triangle meshes of their own heightmap
patch — see `MeshObstacle` in `obstacle_base.py` — so physics and observation
agree by construction; `extra.mesh_stride` coarsens the collision mesh):

- `pitch_ramp` — two opposed ramps with a flat section between; `shape: A`
  (crest) or `shape: V` (trough). Grades the ramp ANGLE (`min_angle`/`max_angle`).
- `diagonal_ramp` — an A ridge whose fall line is yawed from the travel axis
  (pitch and roll at once); yaw grades, `mirror_alternate` flips it.
- `diagonal_stairs` — `raised_stairs` yawed in the lane, one track per riser first.
- `cross_slope` — the lane tilted about the travel axis, twisting in from
  flat so there is no entry step (a helicoid); grades the roll angle.
- `crossing_ramps` — NIST crossing pitch/roll ramps: a triangle wave of
  wedges along X in anti-phase between the two lane halves.
- `wave` — a sinusoidal swell over whole periods, optionally yawed.

Other additions:

- `round_log` — a cylinder lying across the lane (no edge to hook), optionally
  yawed (`min_yaw`/`max_yaw`, the `diagonal_log` row); `diff.height` is its diameter.
- `stepfield` — a NIST/RoboCupRescue stepfield pallet: square posts at 4 height
  levels, adjacent posts never more than 2 levels apart, flat/hill/cross/
  diagonal-hill layouts cycling by repeat — traversable by construction.
- `tilted_pallet` — a pallet-sized slab pitched and yawed, low edge flush with
  the ground (a ramp and a drop that are both off-square). Native tilted box.
- `offset_gate` — a wall with an off-centre opening (`n_gates: 2` = chicane);
  pair with the row's `goal_lateral_offset`. The only type that needs turning.
- `sequence` — several obstacles chained in one tile (`extra.parts`, each with
  its own `width`, listed in the order the robot meets them).
- `stump_field` — three overlapping `stump` bumps staggered left/right along the
  path, growing toward the far end; never flat in between. Positions, widths and
  heights are jittered per tile from the tile seed (`extra.jitter`, 0 = regular).
- `steep_hill` / `twisted_hill` / `bumpy_hill` — long steep ramps (angle AND ramp
  length grade, `n_hills` chains hills that grow along the path) where the body
  must stay pitched: plain, with a sign-changing roll twist, or with seeded random
  bumps and hollows (Gaussian about the drive line). `step_tread: 0.02` builds the
  ramps as 2 cm stairs with vertical risers (`usd_utils.add_surface_mesh`).

Per-row physics (any row's `extra`):

- `friction: 2.0` (or `{static, dynamic}`) — binds a PhysX material with that
  friction to every collider of the row's tiles (combine mode multiply, like the
  env's scene material). Everything else keeps the scene default.
- `edge_lip: true` (or `{friction: 4.0, height: 0.01, width: 0.03, min_rise: 0.04}`)
  — a high-friction nosing box along every rising edge of the tile's heightmap
  (`edges.py`): the stand-in for the real track's protrusions catching a step
  edge, so plain ground can have realistic friction and the robot can still climb.
  Not painted into the heightmap. On a stepped hill the risers become their own
  mesh prim with the lip friction instead.

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

### Row and course options

- `name:` (row) — the env-type label the training side reads from
  `gen_config/` (two `raised_stairs` rows can be `raised_stairs` and
  `steep_stairs`). Defaults to `type`.
- `grade: linear|quadratic|sqrt` (row) — how the repeat index maps to `t`.
- `extra.mirror_alternate: true` (row) — anything with a crossing angle or a
  raised side flips its handedness on odd repeats (`DifficultyParams.mirror_sign`).
- `extra.max_step` (cobblestones) — NIST adjacency rule for random fields.
- `goal_lateral_offset: [min, max]` (row) — target shifted across the lane,
  graded and alternating in sign (follows `offset_gate`'s last opening).
- `bidirectional: true` (course) — a second birth entry per tile driving +X.
- `spawn_lateral_jitter: <m>` (course) — extra entries with the start at ±jitter.
- `include_rows: <other config>` (course) — prepend that course's rows (the
  `mixed_v2_full` course = `mixed_v2_straight` + its turning rows); the
  resolved config is what gets written to `gen_config/`.
- `arena_band: {wall_height, wall_width}` (course) — flat ground filling the
  `border_width` ring plus a fence around it, so overshooting robots stay on ground.
- `feature_offset: {x, y}` (course or row `extra`) — each tile's feature at a seeded
  random offset (built in an off-centre sub-tile with flat filler; nothing leaks
  into neighbours). Holdout only; one-way stairs, sequences and flat lanes are exempt.
- `decor: {...}` / `decor: false` (course) — the eval-only markings written to
  `usd/<name>_decor.usd` (see `ftr_terrain_gen/decor.py`): tile lines, start discs,
  goal squares with hazard frames, start-to-start path lines. The base USD only
  carries `displayColor` (obstacles white, ground grey, walls dark).

Birth entries are emitted in whole-course passes (all tiles forward, all tiles in
reverse, then the same for each jittered start), so the first N robots land on N
different tiles.

Every obstacle exposes tunable defaults (widths, grid size, etc.) via a
row's `extra:` dict in `terrain_config.yaml` — see
`terrain_config_example.yaml` for worked examples, or each obstacle's file
under `ftr_terrain_gen/obstacles/` for its full parameter list.

New types implement the `Obstacle` interface in
`ftr_terrain_gen/obstacle_base.py` (or `MeshObstacle`, implementing only
`build_heightmap`) and register in `ftr_terrain_gen/registry.py`.

## Courses shipped here

- `terrain_config_example.yaml` — `custom_mixed`, the original 17-row course.
- `terrain_config_mixed_v2_straight.yaml` / `_full.yaml` / `_holdout.yaml` — the
  v2 training courses (straight-drive; + turning rows) and the generalization
  course; `terrain_config_marv_calib.yaml` — each obstacle class graded past
  MARV's known limit, for calibration evals. Rationale, the MARV envelope and
  the literature are in `MARV_RL/docs/terrain_v2_design.md`.

## Tests

```bash
pytest tests/
```

Covers grid layout math, heightmap sanity, and a `usd-core` smoke test
(collision APIs, bounding boxes). Physics/visual verification still needs
Isaac Sim.

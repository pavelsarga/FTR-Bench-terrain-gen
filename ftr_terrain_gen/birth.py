"""birth.json generation: one start/target pair PER TILE, placed just inside
that tile's own flat spawn/goal margins (not at the tile center, which is
where the obstacle itself — e.g. raised_platform's platform — sits), so the
robot spawns in front of the obstacle and its target is just past it.

Entries are listed COLUMN-MAJOR: one tile per row (obstacle type) before
moving on to the next column, so a run with fewer robots than tiles spreads
them across all obstacle types instead of filling one type's row first
(`FtrEnv._prepare_reset_info` hands entries out with `itertools.cycle`).
Columns are traversed hardest-repeat-first, and start/target are swapped so
the robot drives in the -X direction (from the tile's trailing edge to its
leading edge) instead of +X. With the augmentations below the list is made
of whole-course PASSES: every tile forward, then every tile in reverse, then
the same again for each jittered start — never several variants of one tile
in a row.

Three optional augmentations, all resolved here at generation time (the
consumer only ever sees a flat list of entries, and `env_type_registry`
locates a tile by rounding the TARGET to the nearest tile centre, so every
entry below still maps to its tile with no consumer change):

* `bidirectional` — a second entry per tile driving +X (start at the
  leading edge, facing yaw 0). RoboCupRescue scores every mobility lane
  end-to-end in BOTH directions, and for one-way features (a staircase, a
  ramp, a pit) the reverse leg is a different obstacle for free.
* `spawn_lateral_jitter` — extra entries with the START shifted +/- this
  many metres across the lane (target unchanged), so the policy does not
  learn that every obstacle is met dead-centre.
* per-row `goal_lateral_offset` [min, max] — the TARGET shifted across the
  lane by an amount graded with the repeat and alternating in sign by
  repeat, matching `offset_gate`'s opening. This is the only entry type
  that requires the robot to turn.
"""

from __future__ import annotations

from ftr_terrain_gen.grid import CourseGrid
from ftr_terrain_gen.offset import resolve_obstacle

FACING_NEGATIVE_X = [0, 0, 3.14]
FACING_POSITIVE_X = [0, 0, 0.0]


def _entry(start, target, forward: bool) -> dict:
    orient = FACING_NEGATIVE_X if forward else FACING_POSITIVE_X
    return {
        "start_point": [float(v) for v in start],
        "start_orient": list(orient),
        "target_point": [float(v) for v in target],
        "target_orient": list(orient),
    }


def build_birth(
    grid: CourseGrid,
    z_clearance: float = 0.125,
    birth_clearance: float = 1.0,
    bidirectional: bool = False,
    spawn_lateral_jitter: float = 0.0,
) -> list[dict]:
    """`birth_clearance` is how far in from each tile's edge (in meters,
    along the travel/X direction) the spawn/target point sits — must be
    smaller than the obstacle's own flat margin (each obstacle centers its
    feature in the tile, so front and back margins are equal) or the point
    will land on/inside the obstacle itself. Default 1.0m, per spec. Every
    obstacle's margin/tile.width is sized so the flat clearance from this
    point to the nearest riser/edge stays the same as it was at the
    previous 0.6m default (MARV is 1.1m long and needs at least 0.6m of
    clearance beyond its own half-length) — moving the point further from
    the tile edge does not eat into that. The per-tile obstacle's
    `birth_offsets()` adds extra Z on top of `z_clearance` for types whose
    start/target sit above ground level.
    """
    base_z = grid.base_z + z_clearance
    half_w = grid.tile_width / 2
    tile_spec = grid.tile_spec()

    # column-major: all rows of a column first; columns hardest-first
    tiles = sorted(grid.iter_tiles(), key=lambda t: (-t.col_index, t.row_index))
    # Entries are emitted in PASSES over the whole course — pass 0: every tile
    # forward; pass 1: every tile in reverse; then the jittered starts, again
    # forward-all then reverse-all. FtrEnv hands the list out in order, so the
    # first N robots land on N DIFFERENT tiles; interleaving the variants per
    # tile put 6 robots on the same obstacle before touching the next one.
    passes: list[list[dict]] = []
    for tile in tiles:
        obstacle = resolve_obstacle(tile.obstacle_type, tile.diff)
        row = grid.rows[tile.row_index]
        start_z_offset, target_z_offset = obstacle.birth_offsets(tile_spec, tile.diff)
        leading_edge = tile.x_center - half_w + birth_clearance
        trailing_edge = tile.x_center + half_w - birth_clearance

        goal_dy = 0.0
        if row.goal_lateral_offset is not None:
            lo, hi = row.goal_lateral_offset
            goal_dy = lo + (hi - lo) * tile.diff.t
            # follow the obstacle's own handedness if it has one (offset_gate's last
            # opening), else alternate by repeat
            final = getattr(obstacle, "final_goal_offset", None)
            sign = (1.0 if final(tile.diff) >= 0 else -1.0) if final else (1.0 if tile.col_index % 2 == 0 else -1.0)
            goal_dy *= sign

        y = tile.y_center
        start_ys = [y]
        if spawn_lateral_jitter > 0:
            start_ys += [y - spawn_lateral_jitter, y + spawn_lateral_jitter]
        k = 0
        for sy in start_ys:
            variants = [_entry(
                (trailing_edge, sy, base_z + start_z_offset),
                (leading_edge, y + goal_dy, base_z + target_z_offset),
                forward=True,
            )]
            if bidirectional:
                # reverse leg: start where the forward target was (behind a gate
                # row's last opening, plus the same lateral jitter), target back
                # on the centreline where the forward leg started
                variants.append(_entry(
                    (leading_edge, y + goal_dy + (sy - y), base_z + target_z_offset),
                    (trailing_edge, y, base_z + start_z_offset),
                    forward=False,
                ))
            for e in variants:
                if k == len(passes):
                    passes.append([])
                passes[k].append(e)
                k += 1
    return [e for p in passes for e in p]

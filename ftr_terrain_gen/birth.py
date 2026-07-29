"""birth.json generation: one start/target pair PER TILE, placed just inside
that tile's own flat spawn/goal margins (not at the tile center, which is
where the obstacle itself — e.g. raised_platform's platform — sits), so the
robot spawns in front of the obstacle and its target is just past it.

Entries are listed hardest-repeat-first per row (REVERSE column order), and
start/target are swapped so the robot drives in the -X direction (from the
tile's trailing edge to its leading edge) instead of +X.
"""

from __future__ import annotations

from ftr_terrain_gen.grid import CourseGrid
from ftr_terrain_gen.registry import get_obstacle


def build_birth(grid: CourseGrid, z_clearance: float = 0.125, birth_clearance: float = 1.0) -> list[dict]:
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
    facing_negative_x = [0, 0, 3.14]
    tile_spec = grid.tile_spec()

    # rows stay in their configured order; only the within-row column
    # (difficulty) traversal is reversed
    tiles = sorted(grid.iter_tiles(), key=lambda t: (t.row_index, -t.col_index))
    entries: list[dict] = []
    for tile in tiles:
        obstacle = get_obstacle(tile.obstacle_type)
        start_z_offset, target_z_offset = obstacle.birth_offsets(tile_spec, tile.diff)
        leading_edge = tile.x_center - half_w + birth_clearance
        trailing_edge = tile.x_center + half_w - birth_clearance
        entries.append(
            {
                "start_point": [trailing_edge, tile.y_center, base_z + start_z_offset],
                "start_orient": facing_negative_x,
                "target_point": [leading_edge, tile.y_center, base_z + target_z_offset],
                "target_orient": facing_negative_x,
            }
        )
    return entries

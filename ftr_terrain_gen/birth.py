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


def build_birth(grid: CourseGrid, z_clearance: float = 0.125, birth_clearance: float = 0.5) -> list[dict]:
    """`birth_clearance` is how far in from each tile's edge (in meters,
    along the travel/X direction) the spawn/target point sits — must be
    smaller than the obstacle's own flat margin (e.g. raised_platform's
    FRONT_MARGIN) or the point will land on/inside the obstacle itself.
    """
    z = grid.base_z + z_clearance
    half_w = grid.tile_width / 2
    facing_negative_x = [0, 0, 3.14]

    # rows stay in their configured order; only the within-row column
    # (difficulty) traversal is reversed
    tiles = sorted(grid.iter_tiles(), key=lambda t: (t.row_index, -t.col_index))
    entries: list[dict] = []
    for tile in tiles:
        leading_edge = tile.x_center - half_w + birth_clearance
        trailing_edge = tile.x_center + half_w - birth_clearance
        entries.append(
            {
                "start_point": [trailing_edge, tile.y_center, z],
                "start_orient": facing_negative_x,
                "target_point": [leading_edge, tile.y_center, z],
                "target_orient": facing_negative_x,
            }
        )
    return entries

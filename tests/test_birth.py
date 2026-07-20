import pytest

from ftr_terrain_gen.birth import build_birth
from ftr_terrain_gen.grid import CourseGrid, RowConfig


def make_grid():
    rows = [RowConfig(type="raised_platform", min_height=0.2, max_height=0.4)]
    return CourseGrid(
        rows=rows, repeats=4, tile_width=5.0, tile_depth=10.0 / 3.0, cell_size=0.05, base_z=0.5,
        border_width=2.0, course_seed=7,
    )


def test_one_entry_per_tile():
    grid = make_grid()
    birth = build_birth(grid)
    assert len(birth) == grid.repeats * grid.n_rows


def test_entries_ordered_hardest_to_easiest_within_a_row():
    # within a row, entries are listed in REVERSE column order (hardest
    # repeat first, easiest last)
    grid = make_grid()
    birth = build_birth(grid)
    xs = [entry["start_point"][0] for entry in birth]
    assert xs == sorted(xs, reverse=True)


def test_start_is_trailing_edge_target_is_leading_edge():
    # robot drives in the -X direction: start (trailing edge, larger x) ->
    # target (leading edge, smaller x)
    grid = make_grid()
    birth = build_birth(grid)
    for entry in birth:
        assert entry["start_point"][0] > entry["target_point"][0]
        assert entry["start_orient"] == [0, 0, 3.14]
        assert entry["target_orient"] == [0, 0, 3.14]


def test_start_and_target_land_outside_the_platform():
    # raised_platform's platform is centered in the tile (PLATFORM_WIDTH
    # wide, FRONT_MARGIN in from each edge) — start/target must NOT fall
    # inside [platform_x - PLATFORM_WIDTH/2, platform_x + PLATFORM_WIDTH/2],
    # which is centered on the tile (x_center), regardless of platform size,
    # as long as birth_clearance < FRONT_MARGIN (the default 0.5 < 1.5 is).
    from ftr_terrain_gen.obstacles.raised_platform import FRONT_MARGIN, PLATFORM_WIDTH

    grid = make_grid()
    birth = build_birth(grid, birth_clearance=0.5)
    # birth.py lists tiles in (row_index, -col_index) order — match that here
    tiles = sorted(grid.iter_tiles(), key=lambda t: (t.row_index, -t.col_index))
    for entry, tile in zip(birth, tiles):
        platform_lo = tile.x_center - PLATFORM_WIDTH / 2
        platform_hi = tile.x_center + PLATFORM_WIDTH / 2
        assert entry["start_point"][0] > platform_hi
        assert entry["target_point"][0] < platform_lo
        # sanity: also confirm they're still within the tile itself
        assert tile.x_center - grid.tile_width / 2 <= entry["target_point"][0]
        assert entry["start_point"][0] <= tile.x_center + grid.tile_width / 2

import numpy as np
import pytest

from ftr_terrain_gen.grid import CourseGrid, RowConfig


def make_grid(**overrides):
    rows = overrides.pop("rows", [RowConfig(type="a", min_height=0.1, max_height=0.5)])
    defaults = dict(
        rows=rows, repeats=4, tile_width=5.0, tile_depth=10.0 / 3.0, cell_size=0.05, base_z=0.5,
        border_width=2.0, course_seed=1,
    )
    defaults.update(overrides)
    return CourseGrid(**defaults)


def test_course_extent():
    grid = make_grid()
    assert grid.course_width_x == pytest.approx(20.0)
    assert grid.course_depth_y == pytest.approx(10.0 / 3.0)


def test_map_bounds_include_border():
    grid = make_grid()
    lower, upper = grid.map_lower, grid.map_upper
    assert lower[0] == pytest.approx(-(20.0 / 2 + 2.0))
    assert upper[0] == pytest.approx(20.0 / 2 + 2.0)


def test_world_to_index_roundtrip():
    # Same `compensation = -(lower/cell_size)` truncation as the real
    # MapHelper — only exact at the lower corner when lower/cell_size is a
    # whole number, so assert "near zero", not "exactly zero".
    grid = make_grid()
    lower = np.array(grid.map_lower[:2])
    i, j = grid.world_to_index((lower[0] + 0.001, lower[1] + 0.001))
    assert abs(i) <= 1 and abs(j) <= 1


def test_difficulty_grades_low_to_high():
    grid = make_grid()
    tiles = sorted(grid.iter_tiles(), key=lambda t: t.col_index)
    heights = [t.diff.height for t in tiles]
    assert heights[0] == pytest.approx(0.1)
    assert heights[-1] == pytest.approx(0.5)
    assert heights == sorted(heights)


def test_every_row_has_the_same_repeat_count():
    rows = [RowConfig(type="a", min_height=0.1, max_height=0.5), RowConfig(type="b", min_height=0.1, max_height=0.5)]
    grid = make_grid(rows=rows, repeats=6)
    for row_index in range(2):
        tiles = [t for t in grid.iter_tiles() if t.row_index == row_index]
        assert len(tiles) == 6
        assert all(t.obstacle_type == rows[row_index].type for t in tiles)


def test_bbox_stays_within_course_width():
    rows = [RowConfig(type="a", min_height=0.1, max_height=0.5)]
    grid = make_grid(rows=rows, repeats=2)
    max_x = max(abs(t.x_center) for t in grid.iter_tiles()) + grid.tile_width / 2
    assert max_x <= grid.course_width_x / 2 + 1e-9


def test_seed_reproducible():
    grid_a = make_grid()
    grid_b = make_grid()
    seeds_a = [t.diff.seed for t in grid_a.iter_tiles()]
    seeds_b = [t.diff.seed for t in grid_b.iter_tiles()]
    assert seeds_a == seeds_b


def test_growing_repeats_does_not_reshuffle_existing_columns_seeds():
    rows = [RowConfig(type="a", min_height=0.1, max_height=0.5), RowConfig(type="b", min_height=0.1, max_height=0.5)]
    grid_a = make_grid(rows=rows, repeats=4)
    grid_b = make_grid(rows=rows, repeats=6)  # course-wide repeats grown
    seeds_a = {(t.row_index, t.col_index): t.diff.seed for t in grid_a.iter_tiles()}
    seeds_b = {(t.row_index, t.col_index): t.diff.seed for t in grid_b.iter_tiles()}
    shared_keys = seeds_a.keys() & seeds_b.keys()
    assert len(shared_keys) == 8  # 2 rows x 4 original columns
    assert {k: seeds_a[k] for k in shared_keys} == {k: seeds_b[k] for k in shared_keys}

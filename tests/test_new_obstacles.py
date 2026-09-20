"""Sloped/analytic obstacles, the mesh primitive, and the birth augmentations."""
import math

import numpy as np
import pytest

from ftr_terrain_gen.birth import build_birth
from ftr_terrain_gen.grid import CourseGrid, RowConfig
from ftr_terrain_gen.obstacle_base import DifficultyParams, TileSpec
from ftr_terrain_gen.registry import REGISTRY, get_obstacle
from ftr_terrain_gen.shapes import paint_tilted_rect

TILE = TileSpec(width=8.8, depth=10.0 / 3.0, cell_size=0.05, base_z=0.5)

# every new type with a representative row config (name -> (min_h, max_h, extra))
NEW_ROWS = {
    "pitch_ramp": (0.0, 0.0, {"min_angle": 8, "max_angle": 30, "shape": "A"}),
    "pitch_ramp_v": (0.0, 0.0, {"min_angle": 8, "max_angle": 30, "shape": "V"}),
    "diagonal_ramp": (0.0, 0.0, {"min_angle": 10, "max_angle": 22, "mirror_alternate": True}),
    "diagonal_stairs": (0.36, 0.54, {"mirror_alternate": True}),
    "cross_slope": (0.0, 0.0, {"min_roll": 5, "max_roll": 20, "mirror_alternate": True}),
    "crossing_ramps": (0.0, 0.0, {"min_angle": 5, "max_angle": 15}),
    "wave": (0.05, 0.20, {"min_yaw": 0, "max_yaw": 25, "mirror_alternate": True}),
    "round_log": (0.15, 0.28, {"min_yaw": 0, "max_yaw": 40, "mirror_alternate": True}),
    "lowered_platform": (0.20, 0.32, {}),
    "stepfield": (0.15, 0.30, {}),
    "tilted_pallet": (0.0, 0.0, {"mirror_alternate": True}),
    "offset_gate": (0.0, 0.0, {"n_gates": 2, "mirror_alternate": True}),
    "steep_hill": (0.0, 0.0, {"min_angle": 20, "max_angle": 38}),
    "steep_hill_v": (0.0, 0.0, {"min_angle": 18, "max_angle": 34, "n_hills": 2, "ramp_length": 1.0, "crest_length": 0.3}),
    "twisted_hill": (0.0, 0.0, {"min_angle": 18, "max_angle": 32, "min_roll": 6, "max_roll": 14, "mirror_alternate": True}),
    "bumpy_hill": (0.0, 0.0, {"min_angle": 18, "max_angle": 32, "min_bump": 0.04, "max_bump": 0.12}),
    "sequence": (0.0, 0.0, {"parts": [
        {"type": "round_log", "min_height": 0.15, "max_height": 0.25, "width": 4.4},
        {"type": "raised_platform", "min_height": 0.2, "max_height": 0.3, "width": 4.4,
         "extra": {"platform_width": 1.0}},
    ]}),
}


def _diff(lo, hi, extra, t=1.0, col=0, seed=11):
    return DifficultyParams(t=t, height=lo + (hi - lo) * t, seed=seed, extra=extra, col_index=col)


def _type(name):
    return name[:-2] if name.endswith("_v") else name


def test_all_new_types_registered():
    for name in NEW_ROWS:
        assert _type(name) in REGISTRY


@pytest.mark.parametrize("name", list(NEW_ROWS))
@pytest.mark.parametrize("col", [0, 1, 9])
def test_heightmap_is_finite_flush_at_edges_and_bounded(name, col):
    lo, hi, extra = NEW_ROWS[name]
    obstacle = get_obstacle(_type(name))
    diff = _diff(lo, hi, extra, t=col / 9, col=col)
    h = obstacle.build_heightmap(TILE, diff)
    assert h.shape == (TILE.nx, TILE.ny)
    assert np.all(np.isfinite(h))
    # spawn/goal margins (1 m in from each X edge, whole lane) are flat ground
    m = int(1.0 / TILE.cell_size)
    assert np.allclose(h[:m, :], TILE.base_z, atol=1e-5), f"{name}: leading margin not flat"
    assert np.allclose(h[-m:, :], TILE.base_z, atol=1e-5), f"{name}: trailing margin not flat"
    # nothing taller than the tallest deliberate feature (1.0 m gate walls; the steep hills
    # reach 1.8 * tan(38 deg) = 1.41 m plus the twist), nothing below the shared floor
    cap = 1.8 if "hill" in name else 1.0
    assert h.max() - TILE.base_z <= cap + 1e-6
    assert h.min() >= TILE.base_z - 0.75


def test_pitch_ramp_angle_and_shape():
    obstacle = get_obstacle("pitch_ramp")
    for shape, sign in (("A", 1), ("V", -1)):
        diff = _diff(0, 0, {"min_angle": 20, "max_angle": 20, "shape": shape, "ramp_length": 1.2, "plateau_length": 1.0})
        h = obstacle.build_heightmap(TILE, diff)
        crest = sign * (h.max() if sign > 0 else h.min())
        assert crest == pytest.approx(sign * (TILE.base_z + sign * 1.2 * math.tan(math.radians(20))), abs=0.02)
        # slope along the centre row equals tan(20 deg) on the ramp
        row = h[:, TILE.ny // 2]
        grad = np.diff(row) / TILE.cell_size
        assert np.max(np.abs(grad)) == pytest.approx(math.tan(math.radians(20)), abs=0.02)


def test_mirror_alternate_flips_handedness():
    obstacle = get_obstacle("diagonal_ramp")
    lo, hi, extra = NEW_ROWS["diagonal_ramp"]
    even = obstacle.build_heightmap(TILE, _diff(lo, hi, extra, t=0.5, col=2))
    odd = obstacle.build_heightmap(TILE, _diff(lo, hi, extra, t=0.5, col=3))
    assert not np.allclose(even, odd)
    assert np.allclose(even, odd[:, ::-1], atol=1e-4)  # mirror image across the lane


def test_cross_slope_is_flush_at_entry_and_tilted_in_the_middle():
    h = get_obstacle("cross_slope").build_heightmap(TILE, _diff(0, 0, {"min_roll": 15, "max_roll": 15}))
    mid = h[TILE.nx // 2, :]
    slope = (mid[-1] - mid[0]) / (TILE.depth - TILE.cell_size)
    assert slope == pytest.approx(math.tan(math.radians(15)), abs=0.01)


def test_crossing_ramps_left_and_right_are_antiphase():
    h = get_obstacle("crossing_ramps").build_heightmap(TILE, _diff(0, 0, {"min_angle": 10, "max_angle": 10}))
    left = h[:, TILE.ny // 4] - TILE.base_z
    right = h[:, 3 * TILE.ny // 4] - TILE.base_z
    assert np.allclose(left, -right, atol=1e-5)
    assert np.abs(left).max() == pytest.approx(0.6 * math.tan(math.radians(10)), abs=0.01)


def test_stepfield_respects_adjacency_rule_for_every_layout():
    obstacle = get_obstacle("stepfield")
    for col in range(4):
        lv = obstacle._levels(_diff(0.15, 0.30, {}, t=1.0, col=col))
        assert lv.min() >= 0 and lv.max() == 3
        assert np.abs(np.diff(lv, axis=0)).max() <= 2
        assert np.abs(np.diff(lv, axis=1)).max() <= 2


def test_cobblestones_max_step_rule():
    obstacle = get_obstacle("cobblestones")
    _, _, _, heights = obstacle._grid(_diff(0.3, 0.3, {"max_step": 0.1, "grid_n": 11}))
    assert np.abs(np.diff(heights, axis=0)).max() <= 0.1 + 1e-9
    assert np.abs(np.diff(heights, axis=1)).max() <= 0.1 + 1e-9


def test_paint_tilted_rect_matches_plane():
    h = np.full((TILE.nx, TILE.ny), TILE.base_z, dtype=np.float32)
    top = (0.0, 0.0, TILE.base_z + 0.2)
    paint_tilted_rect(h, TILE, top, length=1.2, width=1.6, slope_deg=15, yaw_deg=25)
    xs = np.linspace(-TILE.width / 2, TILE.width / 2, TILE.nx, endpoint=False) + TILE.cell_size / 2
    ys = np.linspace(-TILE.depth / 2, TILE.depth / 2, TILE.ny, endpoint=False) + TILE.cell_size / 2
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    u = gx * math.cos(math.radians(25)) + gy * math.sin(math.radians(25))
    inside = h > TILE.base_z + 1e-6
    assert inside.any()
    assert np.allclose(h[inside], TILE.base_z + 0.2 + math.tan(math.radians(15)) * u[inside], atol=1e-5)
    assert np.abs(u[inside]).max() <= 0.6 * math.cos(math.radians(15)) + 1e-6


def test_sequence_places_parts_in_travel_order():
    lo, hi, extra = NEW_ROWS["sequence"]
    h = get_obstacle("sequence").build_heightmap(TILE, _diff(lo, hi, extra, t=1.0))
    # first part (log, +X half) crest = 0.25, second (platform, -X half) = 0.30
    assert h[TILE.nx // 2:, :].max() == pytest.approx(TILE.base_z + 0.25, abs=0.02)
    assert h[: TILE.nx // 2, :].max() == pytest.approx(TILE.base_z + 0.30, abs=1e-5)


def test_grade_curves():
    lin = RowConfig(type="raised_platform", min_height=0, max_height=1)
    sq = RowConfig(type="raised_platform", min_height=0, max_height=1, grade="sqrt")
    assert lin.graded_t(0.25) == 0.25
    assert sq.graded_t(0.25) == 0.5
    with pytest.raises(ValueError):
        RowConfig(type="x", grade="cubic").graded_t(0.5)


def _grid(**kw):
    rows = [
        RowConfig(type="raised_platform", min_height=0.2, max_height=0.3),
        RowConfig(type="offset_gate", extra={"n_gates": 1, "mirror_alternate": True}, goal_lateral_offset=(0.3, 0.5)),
    ]
    return CourseGrid(rows=rows, repeats=4, tile_width=8.8, tile_depth=10 / 3, cell_size=0.05, base_z=0.5, course_seed=1, **kw)


def test_birth_bidirectional_doubles_entries_and_reverses_direction():
    grid = _grid()
    fwd = build_birth(grid)
    both = build_birth(grid, bidirectional=True)
    assert len(both) == 2 * len(fwd)
    rev = [e for e in both if e["start_orient"] == [0, 0, 0.0]]
    assert len(rev) == len(fwd)
    for e in rev:
        assert e["start_point"][0] < e["target_point"][0]  # drives +X


def test_birth_lateral_jitter_and_goal_offset():
    grid = _grid()
    entries = build_birth(grid, spawn_lateral_jitter=0.2)
    assert len(entries) == 3 * grid.repeats * grid.n_rows
    gate_y = (1 - (grid.n_rows - 1) / 2) * grid.tile_depth
    gate = [e for e in entries if abs(e["start_point"][1] - gate_y) < 0.3]
    offsets = sorted({round(e["target_point"][1] - gate_y, 3) for e in gate})
    # graded 0.3 -> 0.5 over 4 repeats, alternating sign with mirror_alternate
    assert offsets == sorted({0.3, -0.367, 0.433, -0.5})
    plain = [e for e in entries if abs(e["start_point"][1] - gate_y) > 0.3]
    assert all(e["target_point"][1] == e["start_point"][1] or abs(e["start_point"][1] - e["target_point"][1]) == pytest.approx(0.2) for e in plain)


pxr = pytest.importorskip("pxr")


def test_mesh_surface_matches_heightmap(tmp_path):
    from pxr import Usd, UsdGeom, UsdPhysics

    from ftr_terrain_gen.assembler import build_usd
    from ftr_terrain_gen.grid import CourseGrid

    rows = [RowConfig(type=_type(n), min_height=lo, max_height=hi, extra=extra) for n, (lo, hi, extra) in NEW_ROWS.items()]
    grid = CourseGrid(rows=rows, repeats=2, tile_width=8.8, tile_depth=10 / 3, cell_size=0.05, base_z=0.5, course_seed=5)
    usd_path = tmp_path / "new.usd"
    build_usd(grid, usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    n_mesh = 0
    for prim in stage.Traverse():
        if prim.IsA(UsdGeom.Mesh):
            n_mesh += 1
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
            assert prim.HasAPI(UsdPhysics.MeshCollisionAPI)
        if prim.IsA(UsdGeom.Cube) or prim.IsA(UsdGeom.Cylinder):
            assert prim.HasAPI(UsdPhysics.CollisionAPI)
    assert n_mesh > 0
    # mesh vertices sit exactly on the heightmap for a mesh obstacle
    tile = grid.tile_spec()
    for placed in grid.iter_tiles():
        if placed.obstacle_type != "pitch_ramp":
            continue
        h = get_obstacle("pitch_ramp").build_heightmap(tile, placed.diff)
        mesh = UsdGeom.Mesh(stage.GetPrimAtPath(f"/World/row_{placed.row_index:02d}_pitch_ramp/tile_{placed.col_index:02d}/surface"))
        pts = np.array(mesh.GetPointsAttr().Get())
        top = pts[pts[:, 2] > tile.base_z - 1.0]  # exclude the skirt ring
        for x, y, z in top[::37]:
            i = min(int((x + tile.width / 2) / tile.cell_size), tile.nx - 1)
            j = min(int((y + tile.depth / 2) / tile.cell_size), tile.ny - 1)
            assert abs(h[i, j] - z) < 1e-4
        break


def test_birth_passes_cover_every_tile_before_repeating_one():
    grid = _grid()
    entries = build_birth(grid, bidirectional=True, spawn_lateral_jitter=0.2)
    n_tiles = grid.repeats * grid.n_rows
    assert len(entries) == 6 * n_tiles
    # pass 0 = every tile forward, pass 1 = every tile reverse, ...
    for p in range(6):
        chunk = entries[p * n_tiles:(p + 1) * n_tiles]
        tiles = {(round(e["target_point"][0] + e["start_point"][0], 3),
                  round(e["target_point"][1] / grid.tile_depth + (grid.n_rows - 1) / 2)) for e in chunk}
        assert len(tiles) == n_tiles
        forward = {e["start_orient"] == [0, 0, 3.14] for e in chunk}
        assert forward == {p % 2 == 0}


def test_stump_field_is_staggered_and_graded():
    obstacle = get_obstacle("stump_field")
    lo = obstacle.build_heightmap(TILE, _diff(0.20, 0.35, {"mirror_alternate": True}, t=0.0, col=0))
    hi = obstacle.build_heightmap(TILE, _diff(0.20, 0.35, {"mirror_alternate": True}, t=1.0, col=1))
    assert lo.max() == pytest.approx(TILE.base_z + 0.20, abs=1e-3)
    assert hi.max() == pytest.approx(TILE.base_z + 0.35, abs=1e-3)
    # the centreline is never flat inside the field: bumps overlap along X
    mid = lo[:, TILE.ny // 2]
    inside = mid > TILE.base_z + 0.02
    i0, i1 = np.nonzero(inside)[0][[0, -1]]
    assert (i1 - i0) * TILE.cell_size > 2.5
    assert np.all(mid[i0:i1] > TILE.base_z + 0.02)


def test_feature_offset_keeps_feature_inside_tile_and_flat_fillers():
    from ftr_terrain_gen.offset import resolve_obstacle

    extra = {"platform_width": 2.0, "feature_offset": {"x": 0.8, "y": 0.4}}
    seen = set()
    for seed in range(6):
        diff = DifficultyParams(t=1.0, height=0.3, seed=seed, extra=extra, col_index=seed)
        obstacle = resolve_obstacle("raised_platform", diff)
        h = obstacle.build_heightmap(TILE, diff)
        raised = np.argwhere(h > TILE.base_z + 1e-6)
        assert raised.size, "platform missing"
        xc = raised[:, 0].mean() * TILE.cell_size - TILE.width / 2
        seen.add(round(xc, 1))
        assert abs(xc) <= 0.8 + 0.05
        # nothing outside the tile, and the 1 m spawn margins stay flat even at max shift
        m = int(1.0 / TILE.cell_size)
        assert np.allclose(h[:m], TILE.base_z) and np.allclose(h[-m:], TILE.base_z)
        assert h.max() == pytest.approx(TILE.base_z + 0.3)
    assert len(seen) > 1, "offsets should differ between tiles"
    # a recessed feature keeps its pit (fillers never cover the sub-tile)
    diff = DifficultyParams(t=1.0, height=0.3, seed=3, extra={"feature_offset": {"x": 0.8}}, col_index=3)
    h = resolve_obstacle("lowered_platform", diff).build_heightmap(TILE, diff)
    assert h.min() == pytest.approx(TILE.base_z - 0.3)


def test_steep_hill_angle_crest_and_repeat_count():
    obstacle = get_obstacle("steep_hill")
    extra = {"min_angle": 20, "max_angle": 38, "ramp_length": 1.8, "crest_length": 0.6}
    h = obstacle.build_heightmap(TILE, _diff(0, 0, extra, t=1.0, col=9))
    mid = h[:, TILE.ny // 2] - TILE.base_z
    assert mid.max() == pytest.approx(1.8 * math.tan(math.radians(38)), abs=0.02)
    slope = np.degrees(np.arctan(np.abs(np.gradient(mid, TILE.cell_size))))
    assert slope.max() == pytest.approx(38, abs=0.5)
    # the crest is flat for ~0.6 m and the whole feature is symmetric about the tile centre
    assert (mid >= mid.max() - 1e-3).sum() * TILE.cell_size == pytest.approx(0.6, abs=0.1)
    assert np.allclose(mid, mid[::-1], atol=1e-4)
    # ramp length grades too: a harder repeat is steeper AND longer, so taller than tan alone
    graded = {"min_angle": 20, "max_angle": 38, "min_ramp_length": 1.4, "max_ramp_length": 1.8}
    easy = obstacle.build_heightmap(TILE, _diff(0, 0, graded, t=0.0))[:, TILE.ny // 2] - TILE.base_z
    assert easy.max() == pytest.approx(1.4 * math.tan(math.radians(20)), abs=0.02)
    # two hills back to back, the second (-X, met last) steeper and taller (hill_growth), with a
    # ground-level trough between them
    two = get_obstacle("steep_hill").build_heightmap(
        TILE, _diff(0, 0, {"min_angle": 18, "max_angle": 34, "n_hills": 2, "hill_growth": 0.7,
                           "ramp_length": 1.0, "crest_length": 0.3, "trough_length": 0.3}, t=1.0))
    mid2 = two[:, TILE.ny // 2] - TILE.base_z
    left, right = mid2[: TILE.nx // 2], mid2[TILE.nx // 2:]
    assert left.max() == pytest.approx(1.0 * math.tan(math.radians(34)), abs=0.02)
    assert right.max() == pytest.approx(1.0 * math.tan(math.radians(34 * 0.7)), abs=0.02)
    trough = mid2[int(4.4 / TILE.cell_size) - 2: int(4.4 / TILE.cell_size) + 2]
    assert np.all(trough == pytest.approx(0.0, abs=1e-4))


def test_twisted_hill_roll_changes_sign_and_mirrors():
    obstacle = get_obstacle("twisted_hill")
    extra = {"min_angle": 18, "max_angle": 32, "min_roll": 6, "max_roll": 14, "mirror_alternate": True}
    h = obstacle.build_heightmap(TILE, _diff(0, 0, extra, t=1.0, col=0))
    # lateral slope (roll) along the centreline: must take both signs inside the feature
    left, right = h[:, TILE.ny // 2 - 5], h[:, TILE.ny // 2 + 5]
    roll = right - left
    assert roll.max() > 0.02 and roll.min() < -0.02
    # the roll is largest where the sine peaks, and it has the promised amplitude
    dy = 10 * TILE.cell_size
    assert np.degrees(np.arctan(np.abs(roll).max() / dy)) == pytest.approx(14, abs=1.0)
    mirrored = obstacle.build_heightmap(TILE, _diff(0, 0, extra, t=1.0, col=1))
    assert np.allclose(mirrored, h[:, ::-1], atol=1e-4)  # odd repeat = the lane mirrored in Y


def test_bumpy_hill_is_seeded_random_and_graded():
    obstacle = get_obstacle("bumpy_hill")
    extra = {"min_angle": 18, "max_angle": 32, "min_bump": 0.04, "max_bump": 0.12}
    ridge = get_obstacle("steep_hill").build_heightmap(TILE, _diff(0, 0, extra, t=1.0))
    a = obstacle.build_heightmap(TILE, _diff(0, 0, extra, t=1.0, seed=1))
    b = obstacle.build_heightmap(TILE, _diff(0, 0, extra, t=1.0, seed=2))
    assert not np.allclose(a, b)  # different seeds, different bumps
    assert np.allclose(a, obstacle.build_heightmap(TILE, _diff(0, 0, extra, t=1.0, seed=1)))  # reproducible
    dev = a - ridge
    assert 0.10 <= dev.max() <= 0.25  # the tallest bump is about the graded height (bumps may overlap)
    assert dev.min() < -0.02  # and there are hollows
    easy = obstacle.build_heightmap(TILE, _diff(0, 0, extra, t=0.0, seed=1)) - \
        get_obstacle("steep_hill").build_heightmap(TILE, _diff(0, 0, extra, t=0.0))
    assert easy.max() < dev.max()


def test_stump_field_is_randomised_per_tile_but_regular_at_zero_jitter():
    obstacle = get_obstacle("stump_field")
    extra = {"mirror_alternate": True}
    a = obstacle.build_heightmap(TILE, _diff(0.20, 0.35, extra, t=0.5, seed=1))
    b = obstacle.build_heightmap(TILE, _diff(0.20, 0.35, extra, t=0.5, seed=2))
    assert not np.allclose(a, b)
    reg = {**extra, "jitter": 0.0}
    r1 = obstacle.build_heightmap(TILE, _diff(0.20, 0.35, reg, t=0.5, seed=1))
    r2 = obstacle.build_heightmap(TILE, _diff(0.20, 0.35, reg, t=0.5, seed=2))
    assert np.allclose(r1, r2)
    # the tallest bump still hits the graded height, and the saddles never return to flat ground
    for seed in range(1, 8):
        h = obstacle.build_heightmap(TILE, _diff(0.20, 0.35, extra, t=0.0, seed=seed))
        assert h.max() == pytest.approx(TILE.base_z + 0.20, abs=1e-3)
        mid = h[:, TILE.ny // 2]
        inside = mid > TILE.base_z + 0.02
        i0, i1 = np.nonzero(inside)[0][[0, -1]]
        assert np.all(mid[i0:i1] > TILE.base_z + 0.01), f"seed {seed}: flat saddle on the centreline"


def test_stepped_hill_has_vertical_risers_in_the_mesh(tmp_path):
    pxr = pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom

    extra = {"min_angle": 20, "max_angle": 38, "ramp_length": 1.8, "crest_length": 0.6, "step_tread": 0.02}
    diff = _diff(0, 0, extra, t=1.0, col=9)
    for name in ("steep_hill", "twisted_hill", "bumpy_hill"):
        obstacle = get_obstacle(name)
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.Xform.Define(stage, "/t")
        obstacle.build_usd(stage, "/t", TILE, diff)
        pts = np.array([tuple(p) for p in UsdGeom.Mesh(stage.GetPrimAtPath("/t/surface")).GetPointsAttr().Get()])
        # every riser is a pair of vertex rows at the same (x, y) with a different z
        xy = np.round(pts[:, :2], 6)
        _, inv, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
        dup = counts[inv] > 1
        z_by_xy = {}
        for (x, y), z in zip(map(tuple, xy[dup]), pts[dup, 2]):
            z_by_xy.setdefault((x, y), set()).add(round(float(z), 5))
        vertical = sum(1 for zs in z_by_xy.values() if len(zs) > 1)
        n_steps = round(1.8 / 0.02)
        assert vertical >= 2 * n_steps * 2, f"{name}: only {vertical} vertical riser columns"
        # riser height on the centreline ~ tread * tan(angle) = 1.46 cm
        top = pts[pts[:, 2] > TILE.base_z - 0.5]  # drop the perimeter skirt
        mid = top[np.abs(top[:, 1] - top[:, 1].min()) < 1e-6]  # the -Y edge row, sorted by x
        mid = mid[np.argsort(mid[:, 0], kind="stable")]
        rises = np.diff(mid[:, 2])
        risers = rises[np.abs(rises) > 1e-4]
        if name == "steep_hill":
            assert np.allclose(np.abs(risers), 0.02 * math.tan(math.radians(38)), atol=2e-3)
    # the observation heightmap is the same staircase sampled at cell centres: monotone up the ramp
    h = get_obstacle("steep_hill").build_heightmap(TILE, diff)[:, TILE.ny // 2]
    up = h[: TILE.nx // 2]
    assert np.all(np.diff(up) >= -1e-6)


def test_edge_lips_follow_rising_edges_only():
    from ftr_terrain_gen.edges import find_lips

    h = np.full((TILE.nx, TILE.ny), TILE.base_z, dtype=np.float32)
    i0, i1 = 60, 100  # a 2 m long, 0.3 m high platform across the whole lane
    h[i0:i1, :] += 0.3
    lips = find_lips(h, TILE, min_rise=0.04, width=0.03)
    assert len(lips) == 2  # one straight riser at each end, each merged into a single box
    xs = sorted(l.x for l in lips)
    x_lo = -TILE.width / 2 + i0 * TILE.cell_size
    x_hi = -TILE.width / 2 + i1 * TILE.cell_size
    assert xs[0] == pytest.approx(x_lo - 0.015, abs=1e-6)  # overhanging the riser face, outside the platform
    assert xs[1] == pytest.approx(x_hi + 0.015, abs=1e-6)
    assert all(l.length == pytest.approx(TILE.ny * TILE.cell_size, abs=1e-6) and l.along == "y" for l in lips)
    assert all(l.z_top == pytest.approx(TILE.base_z + 0.3) for l in lips)
    # a 2 cm bump is below min_rise: no lip
    h2 = np.full((TILE.nx, TILE.ny), TILE.base_z, dtype=np.float32)
    h2[80:90, :] += 0.02
    assert find_lips(h2, TILE, min_rise=0.04, width=0.03) == []
    # stairs: one lip per riser, on the upper tread, with the tread heights
    stairs = get_obstacle("raised_stairs").build_heightmap(
        TILE, _diff(0.4, 0.6, {"n_steps": 4, "step_length": 0.3, "platform_length": 0.6}, t=1.0))
    lips = find_lips(stairs, TILE, min_rise=0.04, width=0.03)
    assert len(lips) == 8  # 4 up + 4 down
    assert sorted({round(l.z_top - TILE.base_z, 3) for l in lips}) == [0.15, 0.3, 0.45, 0.6]


def test_stepped_hill_risers_get_their_own_high_friction_mesh(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    from ftr_terrain_gen.assembler import build_usd
    from ftr_terrain_gen.grid import CourseGrid, RowConfig

    rows = [RowConfig(type="steep_hill", min_height=0, max_height=0,
                      extra={"min_angle": 20, "max_angle": 30, "step_tread": 0.02, "friction": 1.5, "edge_lip": True})]
    grid = CourseGrid(rows=rows, repeats=1, tile_width=8.8, tile_depth=10.0 / 3.0, cell_size=0.05,
                      base_z=0.5, border_width=2.0, course_seed=3)
    usd_path = tmp_path / "hill.usd"
    build_usd(grid, usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    surface = stage.GetPrimAtPath("/World/row_00_steep_hill/tile_00/surface")
    risers = stage.GetPrimAtPath("/World/row_00_steep_hill/tile_00/risers")
    assert surface.IsValid() and risers.IsValid()

    def mu(prim):
        rel = prim.GetRelationship("material:binding:physics")
        return UsdPhysics.MaterialAPI(stage.GetPrimAtPath(rel.GetTargets()[0])).GetStaticFrictionAttr().Get()

    assert mu(surface) == 1.5 and mu(risers) == 3.0  # DEFAULT_LIP friction
    # every riser face is vertical: its three vertices share one x
    pts = UsdGeom.Mesh(risers).GetPointsAttr().Get()
    idx = UsdGeom.Mesh(risers).GetFaceVertexIndicesAttr().Get()
    for f in range(0, len(idx), 3):
        xs = {round(pts[i][0], 6) for i in idx[f:f + 3]}
        assert len(xs) == 1
    n_steps = round(1.8 / 0.02)
    assert len(idx) // 3 >= 2 * 2 * n_steps  # 2 triangles per riser per lane-width strip, both ramps


def test_round_log_ribs_sit_on_the_log_surface(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics

    from ftr_terrain_gen.assembler import build_usd
    from ftr_terrain_gen.grid import CourseGrid, RowConfig

    rows = [RowConfig(type="round_log", min_height=0.28, max_height=0.28,
                      extra={"min_yaw": 30, "max_yaw": 30, "edge_lip": True})]
    grid = CourseGrid(rows=rows, repeats=1, tile_width=8.8, tile_depth=10.0 / 3.0, cell_size=0.05,
                      base_z=0.5, border_width=2.0, course_seed=3)
    usd_path = tmp_path / "log.usd"
    build_usd(grid, usd_path)
    stage = Usd.Stage.Open(str(usd_path))
    ribs = [p for p in stage.Traverse() if "/lips/rib_" in str(p.GetPath())]
    assert len(ribs) >= 10
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    r, cz = 0.14, 0.5 + 0.14
    for prim in ribs:
        rel = prim.GetRelationship("material:binding:physics")
        assert UsdPhysics.MaterialAPI(stage.GetPrimAtPath(rel.GetTargets()[0])).GetStaticFrictionAttr().Get() == 3.0
        c = cache.ComputeWorldBound(prim).ComputeCentroid()
        # the rib centre is ~1.0-1.5 cm outside the log surface, measured in the cross-section plane
        # (the tile sits at world x = 0 for a single-tile course; the log axis is yawed 30 deg)
        dx = c[0] * math.cos(math.radians(30)) + c[1] * math.sin(math.radians(30))
        dist = math.hypot(dx, c[2] - cz)
        assert r - 0.005 <= dist <= r + 0.02, (str(prim.GetPath()), dist)


def test_rotated_tilted_and_sequence_obstacles_get_their_own_lips(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd

    from ftr_terrain_gen.assembler import build_usd
    from ftr_terrain_gen.grid import CourseGrid, RowConfig

    rows = [
        RowConfig(type="diagonal_trunk", min_height=0.25, max_height=0.25, extra={"edge_lip": True, "min_angle": 30, "max_angle": 30}),
        RowConfig(type="tilted_pallet", min_height=0, max_height=0, extra={"edge_lip": True}),
        RowConfig(type="steep_hill", min_height=0, max_height=0, extra={"edge_lip": True, "step_tread": 0.02, "min_angle": 20, "max_angle": 20}),
        RowConfig(type="sequence", min_height=0, max_height=0, extra={"edge_lip": True, "parts": [
            {"type": "round_log", "min_height": 0.2, "max_height": 0.2, "width": 4.4},
            {"type": "raised_platform", "min_height": 0.2, "max_height": 0.2, "width": 4.4, "extra": {"platform_width": 1.0}}]}),
    ]
    grid = CourseGrid(rows=rows, repeats=1, tile_width=8.8, tile_depth=10.0 / 3.0, cell_size=0.05,
                      base_z=0.5, border_width=2.0, course_seed=3)
    build_usd(grid, tmp_path / "lips.usd")
    stage = Usd.Stage.Open(str(tmp_path / "lips.usd"))
    paths = [str(p.GetPath()) for p in stage.Traverse()]
    n = lambda prefix: sum(1 for q in paths if q.startswith(prefix) and "/lips/lip_" in q or (q.startswith(prefix) and "/lips/rib_" in q))
    assert n("/World/row_00_diagonal_trunk") == 4
    assert n("/World/row_01_tilted_pallet") == 4
    assert n("/World/row_02_steep_hill") == 0  # risers carry the friction, no heightmap bars
    assert any("/World/row_02_steep_hill/tile_00/risers" == q for q in paths)
    assert n("/World/row_03_sequence/tile_00/part_0_round_log") >= 8  # ribs on the log
    assert n("/World/row_03_sequence/tile_00/part_1_raised_platform") == 2  # two edges of the step

import pytest

pxr = pytest.importorskip("pxr")
from pxr import Usd, UsdGeom, UsdPhysics  # noqa: E402

from ftr_terrain_gen.assembler import build_usd  # noqa: E402
from ftr_terrain_gen.grid import CourseGrid, RowConfig  # noqa: E402


def make_grid():
    rows = [RowConfig(type="raised_platform", min_height=0.3, max_height=0.5)]
    return CourseGrid(
        rows=rows, repeats=3, tile_width=5.0, tile_depth=10.0 / 3.0, cell_size=0.05, base_z=0.5,
        border_width=2.0, course_seed=3,
    )


def test_generated_usd_has_collision_on_every_leaf(tmp_path):
    grid = make_grid()
    usd_path = tmp_path / "test.usd"
    build_usd(grid, usd_path)

    stage = Usd.Stage.Open(str(usd_path))
    assert stage is not None
    leafs = [p for p in stage.Traverse() if p.IsA(UsdGeom.Cube) or p.IsA(UsdGeom.Mesh)]
    assert len(leafs) > 0
    for prim in leafs:
        assert prim.HasAPI(UsdPhysics.CollisionAPI), f"{prim.GetPath()} missing CollisionAPI"


def test_generated_usd_bbox_matches_layout(tmp_path):
    grid = make_grid()
    usd_path = tmp_path / "test.usd"
    build_usd(grid, usd_path)

    stage = Usd.Stage.Open(str(usd_path))
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
    rng = bbox_cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    mn, mx = rng.GetMin(), rng.GetMax()
    assert mx[0] <= grid.course_width_x / 2 + 0.5
    assert mn[0] >= -grid.course_width_x / 2 - 0.5
    assert mx[1] <= grid.course_depth_y / 2 + 0.5
    assert mn[1] >= -grid.course_depth_y / 2 - 0.5


def test_row_friction_binds_a_physics_material_only_to_that_row(tmp_path):
    from pxr import UsdShade

    rows = [
        RowConfig(type="raised_stairs", min_height=0.4, max_height=0.6, extra={"friction": 2.0}),
        RowConfig(type="steep_hill", min_height=0.0, max_height=0.0,
                  extra={"min_angle": 20, "max_angle": 38, "friction": {"static": 2.5, "dynamic": 2.0}}),
        RowConfig(type="raised_platform", min_height=0.3, max_height=0.5),
    ]
    grid = CourseGrid(rows=rows, repeats=2, tile_width=8.8, tile_depth=10.0 / 3.0, cell_size=0.05,
                      base_z=0.5, border_width=2.0, course_seed=3)
    usd_path = tmp_path / "friction.usd"
    build_usd(grid, usd_path)
    stage = Usd.Stage.Open(str(usd_path))

    def bound_friction(prim):
        rel = prim.GetRelationship("material:binding:physics")
        if not rel or not rel.GetTargets():
            return None
        api = UsdPhysics.MaterialAPI(stage.GetPrimAtPath(rel.GetTargets()[0]))
        return api.GetStaticFrictionAttr().Get(), api.GetDynamicFrictionAttr().Get()

    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        path = str(prim.GetPath())
        f = bound_friction(prim)
        if "row_00_raised_stairs" in path:
            assert f == (2.0, 2.0), path
        elif "row_01_steep_hill" in path:
            assert f == (2.5, 2.0), path
        else:
            assert f is None, path  # everything else keeps the scene default material
    mat = stage.GetPrimAtPath("/World/Looks/friction_s2_d2")
    assert mat.IsValid()
    assert mat.GetAttribute("physxMaterial:frictionCombineMode").Get() == "multiply"

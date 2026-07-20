"""Thin pxr helpers shared by every obstacle class.

Mirrors the proven pattern in FTR-Benchmark's scripts/make_flat_ground_usd.py:
UsdGeom.Cube + UsdPhysics.CollisionAPI (static collider, no RigidBodyAPI) for
box geometry — GPU-cook-friendly, no mesh-cooking warnings.
"""

from __future__ import annotations

from pxr import Gf, Usd, UsdGeom, UsdPhysics

from ftr_terrain_gen.obstacle_base import TileSpec


def apply_collision(prim: Usd.Prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)


# Every "ground segment" cube (base slab, raised feature, or recessed pit
# floor) has its BOTTOM pinned to this one shared floor — `tile.base_z -
# GROUND_SLAB_THICKNESS` — rather than measured as a fixed thickness down
# from its own top. Since `base_z` is the same constant everywhere in a
# course, every segment anywhere (same tile or a neighboring one) ends at
# the exact same absolute Z, so features always overlap/abut cleanly with
# no boolean subtraction needed AND no segment ever protrudes past a
# neighbor's bottom (a raised platform and a recessed trench both simply
# extend down to the same floor, just with different thickness).
# MUST exceed the single deepest configured min_height/max_height/
# extra.trench_depth/etc. ANYWHERE in the course, or a segment that dips
# below this floor gets a real vertical gap to its neighbors instead of the
# intended shared-floor overlap (add_ground_slab raises loudly rather than
# silently clamping, since a silent clamp would produce an invisible gap).
# 2.0m gives generous headroom above every default in this project (all <= 1m).
GROUND_SLAB_THICKNESS = 2.0


def add_ground_slab(
    stage: Usd.Stage,
    path: str,
    x_center: float,
    y_center: float,
    width: float,
    depth: float,
    top_z: float,
    tile: TileSpec,
    rotate_z_deg: float = 0.0,
) -> Usd.Prim:
    """A flat rectangular ground segment: use for a tile's base, a raised
    platform/tier, or a recessed pit floor — the same primitive, only
    `top_z` and the footprint differ. Bottom is pinned to the course-wide
    shared floor (see `GROUND_SLAB_THICKNESS`), not a fixed thickness below
    `top_z`, so segments never protrude past each other. `rotate_z_deg`
    rotates the footprint about its own center (e.g. for a diagonal beam) —
    rotation about Z doesn't affect height, so the top-face-at-`top_z`
    convention and the floor-pinning both still hold exactly.
    """
    floor_z = tile.base_z - GROUND_SLAB_THICKNESS
    thickness = top_z - floor_z
    if thickness < 0.05:
        raise ValueError(
            f"add_ground_slab({path!r}): top_z={top_z:.3f} is at or below the shared "
            f"floor (tile.base_z - GROUND_SLAB_THICKNESS = {floor_z:.3f}). This feature "
            f"needs usd_utils.GROUND_SLAB_THICKNESS raised above "
            f"{tile.base_z - top_z:.3f} — silently clamping here would instead produce "
            f"a real vertical gap between this segment and its neighbors."
        )
    return add_cube(stage, path, (width, depth, thickness), (x_center, y_center, top_z), rotate_z_deg)


def add_cube(
    stage: Usd.Stage,
    path: str,
    size_xyz: tuple[float, float, float],
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotate_z_deg: float = 0.0,
) -> Usd.Prim:
    """A unit UsdGeom.Cube scaled to `size_xyz`, translated so its TOP face
    sits at `translate[2]` (i.e. translate is the desired top-surface
    position, matching make_flat_ground_usd.py's convention), then optionally
    rotated about Z. Collision-enabled, static.
    """
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    sx, sy, sz = size_xyz
    tx, ty, tz = translate
    # unit cube spans [-0.5, 0.5]; shift down by half-height so `tz` is the top face
    center_z = tz - sz / 2.0
    xformable.AddTranslateOp().Set(Gf.Vec3d(tx, ty, center_z))
    if rotate_z_deg:
        xformable.AddRotateZOp().Set(rotate_z_deg)
    xformable.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
    apply_collision(prim)
    return prim


def add_cylinder(
    stage: Usd.Stage,
    path: str,
    radius: float,
    length: float,
    axis: str,
    translate: tuple[float, float, float],
) -> Usd.Prim:
    """A UsdGeom.Cylinder (native PhysX collision primitive, no mesh cooking)
    with its long axis along world `axis` ("X", "Y", or "Z"), centered at
    `translate`. Collision-enabled, static. Not currently used by any
    obstacle (log_crossing switched to a plain box) — kept for future use.
    """
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(length)
    cyl.CreateAxisAttr(axis)
    prim = cyl.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    apply_collision(prim)
    return prim

"""Thin pxr helpers shared by every obstacle class.

Mirrors the proven pattern in FTR-Benchmark's scripts/make_flat_ground_usd.py:
UsdGeom.Cube + UsdPhysics.CollisionAPI (static collider, no RigidBodyAPI) for
box geometry — GPU-cook-friendly, no mesh-cooking warnings.
"""

from __future__ import annotations

from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

from ftr_terrain_gen.obstacle_base import TileSpec


def apply_collision(prim: Usd.Prim) -> None:
    UsdPhysics.CollisionAPI.Apply(prim)


# ── Per-obstacle friction ─────────────────────────────────────────────────────
# A row can set `extra.friction: 2.0` (or `{static: 2.0, dynamic: 1.8}`) and every
# collider of its tiles gets a UsdPhysics material with that friction bound for the
# "physics" purpose; everything else keeps the scene default (the env's
# `terrain_static_friction` / `terrain_dynamic_friction`, 1.2 / 0.9 in the v2 configs).
# The env's scene material uses friction_combine_mode "multiply", so the effective
# wheel-terrain coefficient is wheel (4.0) x terrain: 4.8 on plain ground, 8.0 at 2.0.
def ensure_friction_material(stage, static: float, dynamic: float | None = None):
    dynamic = static if dynamic is None else dynamic
    path = f"/World/Looks/friction_s{static:g}_d{dynamic:g}".replace(".", "p")
    prim = stage.GetPrimAtPath(path)
    material = UsdShade.Material(prim) if prim.IsValid() else UsdShade.Material.Define(stage, path)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(static))
    api.CreateDynamicFrictionAttr().Set(float(dynamic))
    api.CreateRestitutionAttr().Set(0.0)
    # PhysX resolves a pair's combine mode as the "stronger" of the two materials' modes
    # (average < min < multiply < max). The scene default material is "multiply"; a bound
    # material left at its default "average" would give (4.0 + 2.0) / 2 = 3.0 — LESS grip than
    # the 4.0 x 1.2 = 4.8 of plain ground. Author it as multiply explicitly (PhysxMaterialAPI is
    # a codeless applied schema, so no PhysxSchema import is needed to write it).
    prim = material.GetPrim()
    prim.AddAppliedSchema("PhysxMaterialAPI")
    prim.CreateAttribute("physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token).Set("multiply")
    prim.CreateAttribute("physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token).Set("multiply")
    return material


def bind_friction(stage, root_path: str, material) -> int:
    """Bind `material` (physics purpose) to every collider under `root_path`; returns the count."""
    n = 0
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root_path)):
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material, UsdShade.Tokens.weakerThanDescendants, "physics")
            n += 1
    return n


def friction_spec(extra: dict) -> tuple[float, float] | None:
    """(static, dynamic) from a row's `extra.friction`, or None when unset."""
    f = extra.get("friction")
    if f is None:
        return None
    if isinstance(f, dict):
        static = float(f["static"])
        return static, float(f.get("dynamic", static))
    return float(f), float(f)


# ── Display colour ────────────────────────────────────────────────────────────
# Cosmetic only (viewer / GUI evals): `primvars:displayColor` on every prim.
# Ground-level slabs, the arena band and the flat parts of a mesh are grass
# green; obstacle geometry is light red; walls dark grey. The painted markings
# (tile lines, spawn/goal marks, path lines, hazard frames) are built by
# `ftr_terrain_gen/decor.py` as non-colliding decals.
GROUND_COLOR = (0.5, 0.5, 0.5)  # the plain USD grey
FEATURE_COLOR = (0.95, 0.95, 0.95)  # white
WALL_COLOR = (0.35, 0.35, 0.38)
LIP_COLOR = (0.85, 0.55, 0.15)  # edge lips (edges.py): orange, so the high-friction nosings are visible in the GUI
FLAT_TOLERANCE = 0.01  # mesh vertices within this of tile.base_z count as ground


def set_display_color(prim: Usd.Prim, color: tuple[float, float, float]) -> None:
    gprim = UsdGeom.Gprim(prim)
    if gprim:
        gprim.CreateDisplayColorAttr([Gf.Vec3f(*color)])


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
    color = GROUND_COLOR if abs(top_z - tile.base_z) < 1e-6 else FEATURE_COLOR
    return add_cube(stage, path, (width, depth, thickness), (x_center, y_center, top_z), rotate_z_deg, color=color)


def add_cube(
    stage: Usd.Stage,
    path: str,
    size_xyz: tuple[float, float, float],
    translate: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotate_z_deg: float = 0.0,
    collide: bool = True,
    color: tuple[float, float, float] = FEATURE_COLOR,
) -> Usd.Prim:
    """A unit UsdGeom.Cube scaled to `size_xyz`, translated so its TOP face
    sits at `translate[2]` (i.e. translate is the desired top-surface
    position, matching make_flat_ground_usd.py's convention), then optionally
    rotated about Z. Collision-enabled, static — unless `collide=False`
    (a purely visual decal).
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
    if collide:
        apply_collision(prim)
    set_display_color(prim, color)
    return prim


def add_cylinder(
    stage: Usd.Stage,
    path: str,
    radius: float,
    length: float,
    axis: str,
    translate: tuple[float, float, float],
    rotate_z_deg: float = 0.0,
    collide: bool = True,
    color: tuple[float, float, float] = FEATURE_COLOR,
) -> Usd.Prim:
    """A UsdGeom.Cylinder (native PhysX collision primitive, no mesh cooking)
    with its long axis along `axis` ("X", "Y", or "Z"), centered at
    `translate`, optionally turned `rotate_z_deg` about world Z (a log
    lying diagonally across the lane). Collision-enabled, static. Used by
    `round_log`; log_crossing is a plain box.
    """
    cyl = UsdGeom.Cylinder.Define(stage, path)
    cyl.CreateRadiusAttr(radius)
    cyl.CreateHeightAttr(length)
    cyl.CreateAxisAttr(axis)
    prim = cyl.GetPrim()
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(Gf.Vec3d(*translate))
    if rotate_z_deg:
        xformable.AddRotateZOp().Set(rotate_z_deg)
    if collide:
        apply_collision(prim)
    set_display_color(prim, color)
    return prim


def _rotation_columns(yaw_deg: float, slope_deg: float):
    """Basis vectors (u, v, n) of a slab whose local X axis is turned `yaw_deg`
    about world Z and then raised `slope_deg` above the horizontal along that
    turned direction: u = along-slope, v = across-slope (always horizontal),
    n = the slab's top-face normal. Columns of the rotation matrix."""
    import math

    cy, sy = math.cos(math.radians(yaw_deg)), math.sin(math.radians(yaw_deg))
    cs, ss = math.cos(math.radians(slope_deg)), math.sin(math.radians(slope_deg))
    u = (cy * cs, sy * cs, ss)
    v = (-sy, cy, 0.0)
    n = (-cy * ss, -sy * ss, cs)
    return u, v, n


def add_tilted_slab(
    stage: Usd.Stage,
    path: str,
    top_center: tuple[float, float, float],
    length: float,
    width: float,
    slope_deg: float,
    yaw_deg: float = 0.0,
    thickness: float = 0.5,
) -> Usd.Prim:
    """A box whose TOP face is a plane through `top_center`, `length` long
    along its own (yawed) X axis and rising `slope_deg` along it, `width`
    across. Still a native box collider (no mesh cooking). Unlike
    `add_ground_slab` the bottom is NOT pinned to the shared floor — a tilted
    box's underside would poke out of the ground at the low end — so
    `thickness` is measured perpendicular to the top face and the caller
    puts a normal ground slab underneath. The XY footprint of the top face is
    the yaw-rotated rectangle (length*cos(slope)) x width — pair with
    `shapes.paint_tilted_rect` for the heightmap.
    """
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    prim = cube.GetPrim()
    u, v, n = _rotation_columns(yaw_deg, slope_deg)
    cx, cy, cz = top_center
    # box centre = top-face centre pushed half a thickness down the normal
    center = (cx - n[0] * thickness / 2, cy - n[1] * thickness / 2, cz - n[2] * thickness / 2)
    m = Gf.Matrix4d(1.0)
    # rows of a Gf.Matrix4d are the images of the basis vectors (row-vector convention)
    m.SetRow(0, Gf.Vec4d(u[0] * length, u[1] * length, u[2] * length, 0.0))
    m.SetRow(1, Gf.Vec4d(v[0] * width, v[1] * width, v[2] * width, 0.0))
    m.SetRow(2, Gf.Vec4d(n[0] * thickness, n[1] * thickness, n[2] * thickness, 0.0))
    m.SetRow(3, Gf.Vec4d(center[0], center[1], center[2], 1.0))
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTransformOp().Set(m)
    apply_collision(prim)
    set_display_color(prim, FEATURE_COLOR)
    return prim


def add_heightfield_mesh(
    stage: Usd.Stage,
    path: str,
    heights,
    tile: TileSpec,
    stride: int = 1,
    skirt_z: float | None = None,
) -> Usd.Prim:
    """A triangulated UsdGeom.Mesh of a per-tile height array (shape
    (tile.nx, tile.ny), absolute world Z, i.e. exactly what an obstacle's
    `build_heightmap` returns), with one vertex per cell centre (every
    `stride`-th cell) — so the physics surface IS the observation heightmap,
    by construction. Static triangle-mesh collider (MeshCollisionAPI
    approximation "none"). A vertical skirt from the perimeter down to
    `skirt_z` (default: the shared ground floor) closes the sides so a robot
    in a neighbouring lane never sees an overhang; the caller still adds a
    plain ground slab underneath at the patch's minimum height.
    """
    import numpy as np

    h = np.asarray(heights, dtype=float)
    nx, ny = h.shape
    ix = list(range(0, nx, stride))
    iy = list(range(0, ny, stride))
    if ix[-1] != nx - 1:
        ix.append(nx - 1)
    if iy[-1] != ny - 1:
        iy.append(ny - 1)
    xs = [-tile.width / 2 + (i + 0.5) * tile.cell_size for i in ix]
    ys = [-tile.depth / 2 + (j + 0.5) * tile.cell_size for j in iy]
    return add_surface_mesh(stage, path, xs, ys, h[np.ix_(ix, iy)], tile, skirt_z)


def add_surface_mesh(
    stage: Usd.Stage,
    path: str,
    xs,
    ys,
    z,
    tile: TileSpec,
    skirt_z: float | None = None,
    riser_path: str | None = None,
) -> Usd.Prim:
    """The general form of `add_heightfield_mesh`: a triangulated surface over
    an explicit, non-uniform vertex grid — `xs` (len na) x `ys` (len nb) with
    heights `z[a, b]` (absolute world Z). Consecutive equal values in `xs`
    (or `ys`) are allowed and make a VERTICAL face between the two height
    rows: that is how `steep_hill` gets true 90-degree risers, which a
    cell-centred heightfield can never express. The outermost vertices are
    stretched to the tile boundary so adjacent tiles meet, and a perimeter
    skirt down to `skirt_z` closes the sides. With `riser_path`, the vertical
    faces (between two consecutive equal `xs`) go into a SECOND mesh prim at
    that path instead, so they can carry their own physics material — the
    steep hills' 2 cm risers get the edge-lip friction while the treads keep
    the row's (see edges.py).
    """
    import numpy as np

    xs = [float(v) for v in xs]
    ys = [float(v) for v in ys]
    z = np.asarray(z, dtype=float)
    na, nb = len(xs), len(ys)
    assert z.shape == (na, nb), (z.shape, na, nb)
    xs[0], xs[-1] = -tile.width / 2, tile.width / 2
    ys[0], ys[-1] = -tile.depth / 2, tile.depth / 2
    if skirt_z is None:
        skirt_z = tile.base_z - GROUND_SLAB_THICKNESS

    points: list[Gf.Vec3f] = []
    index = {}
    for a in range(na):
        for b in range(nb):
            index[(a, b)] = len(points)
            points.append(Gf.Vec3f(xs[a], ys[b], float(z[a, b])))
    counts: list[int] = []
    indices: list[int] = []
    riser_indices: list[int] = []
    for a in range(na - 1):
        vertical = riser_path is not None and abs(xs[a + 1] - xs[a]) < 1e-9
        target = riser_indices if vertical else indices
        for b in range(nb - 1):
            p00, p10, p01, p11 = index[(a, b)], index[(a + 1, b)], index[(a, b + 1)], index[(a + 1, b + 1)]
            # counter-clockwise seen from +Z -> normal up (a vertical riser gets its normal
            # from the winding too: -X facing for a rise toward +X, which is the outward side)
            target += [p00, p10, p11, p00, p11, p01]
            if not vertical:
                counts += [3, 3]

    # perimeter skirt: a ring of vertices at skirt_z under the boundary ring
    def skirt(ring):
        base = len(points)
        for pi in ring:
            p = points[pi]
            points.append(Gf.Vec3f(p[0], p[1], skirt_z))
        for k in range(len(ring) - 1):
            t0, t1 = ring[k], ring[k + 1]
            b0, b1 = base + k, base + k + 1
            indices.extend([t0, b0, b1])
            counts.append(3)
            indices.extend([t0, b1, t1])
            counts.append(3)

    # boundary ring, ordered so the skirt faces point outward: -Y edge (a up),
    # +X edge (b up), +Y edge (a down), -X edge (b down)
    ring = [index[(a, 0)] for a in range(na)]
    ring += [index[(na - 1, b)] for b in range(1, nb)]
    ring += [index[(a, nb - 1)] for a in range(na - 2, -1, -1)]
    ring += [index[(0, b)] for b in range(nb - 2, -1, -1)]
    skirt(ring)

    def _mesh(mesh_path, idx, cnt, color_fn):
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.CreatePointsAttr(points)
        mesh.CreateFaceVertexCountsAttr(cnt)
        mesh.CreateFaceVertexIndicesAttr(idx)
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDoubleSidedAttr(True)
        UsdGeom.Primvar(mesh.CreateDisplayColorAttr([Gf.Vec3f(*color_fn(p)) for p in points])).SetInterpolation(
            UsdGeom.Tokens.vertex)
        prim = mesh.GetPrim()
        apply_collision(prim)
        UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(UsdPhysics.Tokens.none)
        return prim

    # per-vertex colour: ground colour where the surface is at ground level, feature colour elsewhere
    prim = _mesh(path, indices, counts,
                 lambda p: GROUND_COLOR if abs(p[2] - tile.base_z) < FLAT_TOLERANCE else FEATURE_COLOR)
    if riser_indices:
        _mesh(riser_path, riser_indices, [3] * (len(riser_indices) // 3), lambda p: LIP_COLOR)
    return prim


def add_ribbon(
    stage: Usd.Stage,
    path: str,
    points_xyz,
    width: float,
    color: tuple[float, float, float],
    segment_colors=None,
) -> Usd.Prim:
    """A flat ribbon `width` wide along a polyline of (x, y, z) points — a
    painted line on the ground that follows the terrain. Visual only: no
    collision. Used for the tile boundaries and the start->target path lines.
    `segment_colors` (one RGB per polyline segment) paints the ribbon
    piecewise, e.g. alternating hazard stripes, in a single prim."""
    import numpy as np

    pts = np.asarray(points_xyz, dtype=float)
    if len(pts) < 2:
        raise ValueError("add_ribbon needs at least two points")
    d = np.gradient(pts[:, :2], axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], axis=1)
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
    left = pts[:, :2] + n * width / 2
    right = pts[:, :2] - n * width / 2
    verts: list[Gf.Vec3f] = []
    for k in range(len(pts)):
        verts.append(Gf.Vec3f(float(left[k, 0]), float(left[k, 1]), float(pts[k, 2])))
        verts.append(Gf.Vec3f(float(right[k, 0]), float(right[k, 1]), float(pts[k, 2])))
    counts: list[int] = []
    indices: list[int] = []
    for k in range(len(pts) - 1):
        a, b, c, e = 2 * k, 2 * k + 1, 2 * k + 2, 2 * k + 3
        indices += [a, b, e, a, e, c]
        counts += [3, 3]
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr(verts)
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)
    prim = mesh.GetPrim()
    if segment_colors is not None:
        faces = [Gf.Vec3f(*c) for c in segment_colors for _ in range(2)]  # two triangles per segment
        UsdGeom.Primvar(mesh.CreateDisplayColorAttr(faces)).SetInterpolation(UsdGeom.Tokens.uniform)
    else:
        set_display_color(prim, color)
    return prim

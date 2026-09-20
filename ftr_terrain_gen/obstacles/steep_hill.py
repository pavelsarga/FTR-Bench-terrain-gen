"""Steep hills: the STABILITY family.

Every other slope in the course is shallow enough that a policy can treat
"body pitched" as "something is wrong" and try to level out. These hills are
steep enough (up to ~40 deg) and long enough (the whole 1.12 m track sits on
the slope) that the body *must* stay pitched for seconds at a time — the
skill to learn is keeping the robot stable on the slope (weight over the
tracks, rear flippers behind the CoM so it cannot tip over backwards, a
controlled crest transition) rather than fighting the tilt itself.

Three variants, each one adding a disturbance the previous one lacks:

* `steep_hill`   — clean up-ramp, short crest, down-ramp (`n_hills` of them
                   back to back, like the up/down stairs); pitch only.
* `twisted_hill` — the same hill with a roll that CHANGES along the path
                   (helicoidal twist, sign alternates), so the two tracks are
                   at different heights and that difference keeps reversing.
* `bumpy_hill`   — the same hill with random smooth bumps and hollows on the
                   slope (seeded per tile), so the pitch/roll disturbances are
                   irregular and cannot be timed.

The grading knobs are the ramp ANGLE (`min_angle`/`max_angle`, degrees) and
the ramp LENGTH (`min_ramp_length`/`max_ramp_length`, metres), both lerped
by `diff.t`, so a harder repeat is both steeper and taller (height = length
x tan(angle) grows faster than either alone). With `n_hills` >= 2 the hills
also grow ALONG the path (`hill_growth`: the first hill met is that fraction
of the angle, the last is the full angle). `diff.height` is only used when
no angle range is given.

Surface texture: with `step_tread` > 0 the ramps are built as a staircase of
tiny steps — `step_tread` long, `step_tread x tan(angle)` high, with TRUE
VERTICAL risers in the collision mesh (built on a non-uniform vertex grid,
`usd_utils.add_surface_mesh`, not the cell-centred heightfield). A smooth
mesh slope gives the wheels nothing but the friction coefficient to push
against, and PhysX's patch friction turns very high coefficients into
stick-slip instead of grip; a riser gives every wheel a normal contact that
points along the slope. The observation heightmap (5 cm cells) only sees the
average slope of steps this small, which is what the real robot's elevation
map would see too.
"""

from __future__ import annotations

import math

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, MeshObstacle, TileSpec
from ftr_terrain_gen.shapes import cell_centers

RAMP_LENGTH = 1.8  # horizontal run of each ramp (m); longer than the 1.12 m track so the whole robot is on the slope
# (`min_ramp_length`/`max_ramp_length` grade it instead; the v2 course uses 1.4 -> 1.8)
HILL_GROWTH = 0.7  # n_hills >= 2: the first hill met (+X end) has this fraction of the graded angle, the last the full angle
CREST_LENGTH = 0.6  # flat crest between the up- and down-ramp: shorter than the robot, so the crest is a transition, not a rest
TROUGH_LENGTH = 0.3  # flat floor between two consecutive hills (n_hills >= 2)
MIN_ANGLE = 20.0
MAX_ANGLE = 38.0
SHAPE = "A"  # "A": hill(s) above ground. "V": the same profile dug into the ground.
N_HILLS = 1
STEP_TREAD = 0.0  # (m) > 0: terrace the ramps into steps this long (rise = tread * tan(angle)) with vertical risers
FOOT_FADE = 0.5  # (m) the bump term fades to zero over this distance from each foot, so the entry is flush
ROLL_FADE = 1.0  # (m) the roll term does the same over a longer run (smoothstep), so the lane edges do not ramp up abruptly
_EPS = 1e-4  # (m) riser vertices are evaluated this far to either side of the riser

# twisted_hill
MIN_ROLL = 6.0  # roll amplitude (deg) at the easiest repeat
MAX_ROLL = 16.0
TWIST_PERIODS = 1.5  # full sine periods of roll over the feature length: 1.5 -> the sign changes twice

# bumpy_hill
N_BUMPS = 30  # random bumps/hollows over the feature footprint
BUMP_BAND = 1.45  # (m) bump centres are clipped to |y| <= this — nearly the whole 3.33 m lane, so the
# bumps cover the drive line AND both sides of it and cannot be driven around (the lane edge itself
# is left alone so the tile boundary stays clean)
BUMP_SIGMA = 0.6  # (m) lateral bump positions are Gaussian about the drive line with this sigma (0 = uniform
# over the band): ~2/3 of the bumps land within +-0.6 m, where the tracks are, the rest cover the sides
MIN_BUMP = 0.04  # bump height (m) at the easiest repeat (peak of the tallest bump)
MAX_BUMP = 0.12
BUMP_RADIUS = (0.25, 0.45)  # (m) footprint radius range of a bump; a track is ~0.1 m wide, the robot 0.57
HOLLOW_FRACTION = 0.3  # this share of the bumps are hollows (negative), at 60 % of the bump height


class SteepHill(MeshObstacle):
    """`n_hills` steep A-shaped hills back to back, spanning the lane. Each
    hill is an up-ramp of `ramp_length`, a `crest_length` flat and a
    down-ramp of the same angle; hills are separated by a `trough_length`
    flat. The angle grades `min_angle` -> `max_angle` (degrees); `shape: V`
    digs the same profile into the ground instead. The feature is centred in
    the tile, so both spawn margins stay flat.

    A long steep ramp is what makes the robot's stability, not its reach, the
    limit: with the rear flippers flat on a 38 deg slope the static tip-back
    margin is still large (the support polygon reaches 0.56 m behind the
    CoM), but lifting the rear flippers on the climb or hitting the crest at
    full throttle is what tips it. Nothing in the pitch_ramp rows (<= 30 deg,
    1.2 m ramps) ever puts the whole robot on the slope at once.
    """

    name = "steep_hill"

    # -- geometry shared by the three variants -------------------------------------
    def _geometry(self, diff: DifficultyParams):
        if "min_ramp_length" in diff.extra or "max_ramp_length" in diff.extra:
            ramp = diff.lerp_extra("min_ramp_length", "max_ramp_length", RAMP_LENGTH)
        else:
            ramp = diff.extra.get("ramp_length", RAMP_LENGTH)
        crest = diff.extra.get("crest_length", CREST_LENGTH)
        trough = diff.extra.get("trough_length", TROUGH_LENGTH)
        n = int(diff.extra.get("n_hills", N_HILLS))
        if "min_angle" in diff.extra or "max_angle" in diff.extra:
            angle = diff.lerp_extra("min_angle", "max_angle")
        elif diff.height > 0:
            angle = math.degrees(math.atan(diff.height / ramp))
        else:
            angle = MIN_ANGLE + (MAX_ANGLE - MIN_ANGLE) * diff.t
        growth = diff.extra.get("hill_growth", HILL_GROWTH)
        # per-hill angle, indexed from the -X end (met LAST when driving -X): full angle there,
        # `growth` x angle at the +X end — same convention as stump_field's height_ramp
        angles = [angle if n == 1 else angle * (1.0 - (1.0 - growth) * k / (n - 1)) for k in range(n)]
        heights = np.array([ramp * math.tan(math.radians(a)) for a in angles])
        sign = -1.0 if str(diff.extra.get("shape", SHAPE)).upper() == "V" else 1.0
        hill_len = 2 * ramp + crest
        total = n * hill_len + (n - 1) * trough
        tread = float(diff.extra.get("step_tread", STEP_TREAD))
        n_steps = max(1, round(ramp / tread)) if tread > 0 else 0  # steps per ramp; the tread is ramp / n_steps exactly
        return ramp, crest, trough, n, heights, sign, hill_len, total, n_steps

    def ridge(self, gx: np.ndarray, diff: DifficultyParams) -> tuple[np.ndarray, np.ndarray]:
        """Signed hill profile (relative to ground) at every point of `gx`, and
        the distance of each point INSIDE the feature footprint (0 outside),
        which the variants use to fade their extra terms at the feet."""
        ramp, crest, trough, n, heights, sign, hill_len, total, n_steps = self._geometry(diff)
        x = gx + total / 2  # 0 at the -X foot, `total` at the +X foot
        inside = np.clip(np.minimum(x, total - x), 0.0, None)
        # which hill, and the position within its repeating (hill + trough) cell
        period = hill_len + trough
        k = np.clip(np.floor(x / period), 0, n - 1).astype(int)
        xin = x - k * period
        xin = np.where((x < 0) | (x > total), -1.0, xin)  # outside the feature
        # distance from the crest edge outward (negative on the crest), and from the foot inward
        d = np.abs(xin - hill_len / 2) - crest / 2
        u = np.clip(ramp - d, 0.0, ramp)  # 0 at the foot, `ramp` at the crest edge
        if n_steps > 0:
            tread = ramp / n_steps
            frac = np.minimum(np.floor(u / tread + 1e-9), n_steps) / n_steps  # tread k sits at k rises
        else:
            frac = u / ramp
        frac = np.where(xin < 0, 0.0, frac)
        frac = np.where(xin > hill_len, 0.0, frac)  # in a trough
        return sign * heights[k] * frac, inside

    def riser_positions(self, diff: DifficultyParams) -> np.ndarray:
        """Tile-local X of every vertical riser (empty when `step_tread` is 0)."""
        ramp, crest, trough, n, heights, sign, hill_len, total, n_steps = self._geometry(diff)
        if n_steps == 0:
            return np.zeros(0)
        tread = ramp / n_steps
        out = []
        for k in range(n):
            foot_lo = -total / 2 + k * (hill_len + trough)  # -X foot of hill k
            foot_hi = foot_lo + hill_len  # +X foot
            for s in range(1, n_steps + 1):  # the last riser (s = n_steps) reaches the crest
                out.append(foot_lo + s * tread)
                out.append(foot_hi - s * tread)
        return np.array(sorted(out))

    # -- surface --------------------------------------------------------------------
    def surface(self, gx: np.ndarray, gy: np.ndarray, diff: DifficultyParams) -> np.ndarray:
        """Height above ground at arbitrary tile-local points (gx, gy)."""
        z, _ = self.ridge(gx, diff)
        return z

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        gx, gy = cell_centers(tile)
        return (tile.base_z + self.surface(gx, gy, diff)).astype(np.float32)

    def mesh_y_stride(self, tile: TileSpec, diff: DifficultyParams) -> int:
        """Cells per mesh vertex across the lane: the plain hill is constant in Y."""
        return tile.ny  # two vertices: both lane edges

    def build_lips(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams, spec) -> int | None:
        # the stepped mesh carries its own risers (bound to the lip friction by the assembler);
        # the heightmap finder would add bars wherever the twist or a bump makes a 4 cm cell
        # step, which is not an edge at all
        return 0

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        from ftr_terrain_gen.usd_utils import add_ground_slab, add_surface_mesh

        risers = self.riser_positions(diff)
        # vertex X positions: the regular cell centres, plus each riser twice (evaluated a hair
        # to either side, so the two vertex rows sit at the lower and the upper tread height
        # and the face between them is vertical)
        xc = np.linspace(-tile.width / 2, tile.width / 2, tile.nx, endpoint=False) + tile.width / (2 * tile.nx)
        xv = list(xc) + list(risers) + list(risers)
        xe = list(xc) + list(risers - _EPS) + list(risers + _EPS)
        order = np.argsort(xe, kind="stable")
        xv, xe = np.asarray(xv)[order], np.asarray(xe)[order]
        stride = int(diff.extra.get("mesh_stride", 1)) if risers.size == 0 else self.mesh_y_stride(tile, diff)
        jy = list(range(0, tile.ny, max(1, stride)))
        if jy[-1] != tile.ny - 1:
            jy.append(tile.ny - 1)
        yv = np.array([-tile.depth / 2 + (j + 0.5) * tile.cell_size for j in jy])
        gx, gy = np.meshgrid(xe, yv, indexing="ij")
        z = tile.base_z + self.surface(gx, gy, diff)
        h = self.build_heightmap(tile, diff)
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth,
                        float(min(h.min(), z.min())), tile)
        # with `edge_lip`, the vertical risers become their own prim ("risers") so the assembler
        # can give them the lip friction while the treads keep the row's: a small grip area
        # per step, like the track protrusions catching the edges of real stairs
        riser_path = f"{prim_path}/risers" if diff.extra.get("edge_lip") else None
        add_surface_mesh(stage, f"{prim_path}/surface", xv, yv, z, tile, riser_path=riser_path)


class TwistedHill(SteepHill):
    """`steep_hill` whose surface also rolls about the travel axis, with a
    roll angle that varies along the path as a sine: roll(x) = R sin(2 pi
    x / lambda), `twist_periods` periods over the feature, so the high side
    swaps left <-> right (twice, at the default 1.5 periods) while the robot
    is still on the steep slope. R grades `min_roll` -> `max_roll` (degrees);
    the roll term fades in over `roll_fade` (smoothstep) at both feet so the
    entry is flush. `extra.mirror_alternate` flips the phase on odd repeats,
    which together with bidirectional driving shows every roll sequence in
    both handednesses. The roll is linear in Y, so the mesh still needs only
    the two lane-edge vertex rows.
    """

    name = "twisted_hill"

    def surface(self, gx: np.ndarray, gy: np.ndarray, diff: DifficultyParams) -> np.ndarray:
        z, inside = self.ridge(gx, diff)
        total = self._geometry(diff)[7]
        roll_amp = diff.lerp_extra("min_roll", "max_roll") if ("min_roll" in diff.extra or "max_roll" in diff.extra) \
            else MIN_ROLL + (MAX_ROLL - MIN_ROLL) * diff.t
        periods = diff.extra.get("twist_periods", TWIST_PERIODS)
        fade = diff.extra.get("roll_fade", ROLL_FADE)
        s = (gx + total / 2) / total  # 0..1 along the feature
        roll = np.radians(roll_amp) * np.sin(2 * np.pi * periods * s) * diff.mirror_sign
        e = np.clip(inside / fade, 0.0, 1.0)
        envelope = e * e * (3.0 - 2.0 * e)  # smoothstep
        return z + np.tan(roll) * gy * envelope


def _cos_bump(d2: np.ndarray, r: float) -> np.ndarray:
    """Compact, C1-smooth radial bump: cos^2(pi/2 * d/r) inside radius r, 0 outside."""
    d = np.sqrt(d2)
    return np.where(d < r, np.cos(0.5 * np.pi * np.minimum(d, r) / r) ** 2, 0.0)


class BumpyHill(SteepHill):
    """`steep_hill` with `n_bumps` random smooth bumps (and a share of
    hollows, `hollow_fraction`) scattered over the slope, Gaussian about the
    drive line (`bump_sigma` 0.6 m, clipped to `bump_band` |y| <= 1.45 m so
    the sides are covered too) — positions, radii (`bump_radius`
    range) and heights are drawn from the tile's own seed, so
    no two repeats and no two courses share a layout, and the disturbances
    cannot be timed. The tallest bump peaks at `min_bump` -> `max_bump`
    (metres, graded); the rest are 40-100 % of that. Bumps are summed, so
    overlapping ones make irregular shapes, and they fade out over
    `foot_fade` at both feet. The mesh keeps 5 cm vertex rows across the lane
    (`mesh_stride`, default 1) because the bumps are only a few cells wide.
    """

    name = "bumpy_hill"

    def _bumps(self, tile_depth: float, diff: DifficultyParams):
        """Deterministic bump set for this tile: (cx, cy, r, signed height, fade)."""
        total = self._geometry(diff)[7]
        rng = diff.rng()
        n = int(diff.extra.get("n_bumps", N_BUMPS))
        amp = diff.lerp_extra("min_bump", "max_bump") if ("min_bump" in diff.extra or "max_bump" in diff.extra) \
            else MIN_BUMP + (MAX_BUMP - MIN_BUMP) * diff.t
        r_lo, r_hi = diff.extra.get("bump_radius", BUMP_RADIUS)
        hollow_frac = diff.extra.get("hollow_fraction", HOLLOW_FRACTION)
        fade = diff.extra.get("foot_fade", FOOT_FADE)
        half_x = total / 2 - fade
        half_y = min(diff.extra.get("bump_band", BUMP_BAND), tile_depth / 2)
        cx = rng.uniform(-half_x, half_x, size=n)
        sigma = float(diff.extra.get("bump_sigma", BUMP_SIGMA))
        cy = np.clip(rng.normal(0.0, sigma, size=n), -half_y, half_y) if sigma > 0 else rng.uniform(-half_y, half_y, size=n)
        rr = rng.uniform(r_lo, r_hi, size=n)
        hh = rng.uniform(0.4, 1.0, size=n)
        hh[0] = 1.0  # the tallest bump is exactly the graded height
        hollow = rng.uniform(size=n) < hollow_frac
        hollow[0] = False
        a = amp * hh * np.where(hollow, -0.6, 1.0)
        return cx, cy, rr, a, fade

    # the bump band is clipped to the lane, so `surface` needs the tile depth: both entry
    # points stash it before evaluating
    _tile_depth = 10.0 / 3.0

    def surface(self, gx: np.ndarray, gy: np.ndarray, diff: DifficultyParams) -> np.ndarray:
        z, inside = self.ridge(gx, diff)
        cx, cy, rr, a, fade = self._bumps(self._tile_depth, diff)
        bumps = np.zeros_like(gx, dtype=float)
        for k in range(len(cx)):
            bumps += a[k] * _cos_bump((gx - cx[k]) ** 2 + (gy - cy[k]) ** 2, rr[k])
        envelope = np.clip(inside / fade, 0.0, 1.0)
        return z + bumps * envelope

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        self._tile_depth = tile.depth
        return super().build_heightmap(tile, diff)

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        self._tile_depth = tile.depth
        super().build_usd(stage, prim_path, tile, diff)

    def mesh_y_stride(self, tile: TileSpec, diff: DifficultyParams) -> int:
        return int(diff.extra.get("mesh_stride", 1))

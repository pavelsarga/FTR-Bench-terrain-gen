"""extra.lowered (mirrored obstacle), extra.sample (per-tile parameter draws) and raised_platform's
two-tier exit_height_ratio — ftr_terrain_gen/variants.py."""
import numpy as np
import pytest

from ftr_terrain_gen.obstacle_base import DifficultyParams, TileSpec
from ftr_terrain_gen.offset import resolve_obstacle
from ftr_terrain_gen.registry import get_obstacle
from ftr_terrain_gen.variants import LoweredObstacle, SampledObstacle

TILE = TileSpec(width=8.8, depth=10.0 / 3.0, cell_size=0.05, base_z=0.5)
LOWERABLE = {
    "cobblestones": {"grid_n": 11, "max_step": 0.12},
    "half_platform": {},
    "stump_field": {"mirror_alternate": True},
    "wave": {"min_yaw": 0, "max_yaw": 25},
    "stepfield": {},
    "raised_platform": {},
}


def diff(extra, seed=7, height=0.25, col=0):
    return DifficultyParams(t=0.5, height=height, seed=seed, extra=extra, col_index=col)


@pytest.mark.parametrize("name", sorted(LOWERABLE))
def test_lowered_is_the_mirror_about_ground(name):
    base = get_obstacle(name).build_heightmap(TILE, diff(LOWERABLE[name]))
    low = resolve_obstacle(name, diff({**LOWERABLE[name], "lowered": True}))
    assert isinstance(low, LoweredObstacle)
    h = low.build_heightmap(TILE, diff({**LOWERABLE[name], "lowered": True}))
    np.testing.assert_allclose(h - TILE.base_z, -(base - TILE.base_z), atol=1e-6)
    assert h.min() < TILE.base_z - 0.05  # it really dips
    if name == "wave":
        # a sine has crests and troughs either way; mirrored, the first deviation met is a trough
        line = h[:, h.shape[1] // 2] - TILE.base_z
        assert line[np.nonzero(np.abs(line) > 0.01)[0][0]] < 0
    else:
        assert h.max() <= TILE.base_z + 1e-6 < base.max()  # nothing above ground any more


def test_lowered_platform_edge_is_met_going_down_first():
    """Along the centreline in +X the first sharp height change of a lowered raised_platform is a drop."""
    h = resolve_obstacle("raised_platform", diff({"lowered": True})).build_heightmap(TILE, diff({"lowered": True}))
    line = h[:, h.shape[1] // 2]
    dz = np.diff(line)
    first = dz[np.nonzero(np.abs(dz) > 0.06)[0][0]]
    assert first < 0


def test_lowered_birth_offsets_are_negated():
    class Fake:
        name = "fake"

        def birth_offsets(self, tile, d):
            return 0.3, -0.1

    assert LoweredObstacle(Fake()).birth_offsets(TILE, diff({})) == (-0.3, 0.1)


def test_sample_draws_per_tile_in_range_and_deterministic():
    spec = {"sample": {"platform_width": [1.0, 2.6], "exit_height_ratio": [0.6, 1.4]}}
    draws = [SampledObstacle.sampled_diff(diff(spec, seed=s)).extra for s in range(50)]
    w = np.array([d["platform_width"] for d in draws])
    r = np.array([d["exit_height_ratio"] for d in draws])
    assert w.min() >= 1.0 - 1e-9 and w.max() <= 2.6 + 1e-9 and w.std() > 0.2
    assert np.allclose(np.round(w / 0.05) * 0.05, w)  # lengths snapped to the cell
    assert r.min() >= 0.6 and r.max() <= 1.4 and not np.allclose(np.round(r / 0.05) * 0.05, r)  # ratios not snapped
    again = SampledObstacle.sampled_diff(diff(spec, seed=3)).extra
    assert again == draws[3]


def test_rows_without_variants_are_unchanged():
    for name, extra in LOWERABLE.items():
        assert type(resolve_obstacle(name, diff(extra))) is type(get_obstacle(name))


def test_two_tier_platform_drop_differs_from_rise():
    d = diff({"exit_height_ratio": 0.5}, height=0.3)
    h = get_obstacle("raised_platform").build_heightmap(TILE, d)
    line = h[:, h.shape[1] // 2] - TILE.base_z
    dz = np.diff(line)
    edges = dz[np.abs(dz) > 0.06]
    assert edges[0] == pytest.approx(0.3, abs=1e-6)       # climb h
    assert edges[-1] == pytest.approx(-0.15, abs=1e-6)    # drop r h
    assert sorted(set(np.round(line.astype(float), 3).tolist())) == pytest.approx([0.0, 0.15, 0.3])


def test_two_tier_default_is_the_single_platform():
    a = get_obstacle("raised_platform").build_heightmap(TILE, diff({}))
    b = get_obstacle("raised_platform").build_heightmap(TILE, diff({"exit_height_ratio": 1.0}))
    np.testing.assert_array_equal(a, b)

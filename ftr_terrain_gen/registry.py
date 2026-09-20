from __future__ import annotations

from ftr_terrain_gen.obstacle_base import Obstacle
from ftr_terrain_gen.obstacles.cobblestones import Cobblestones
from ftr_terrain_gen.obstacles.cross_slope import CrossSlope
from ftr_terrain_gen.obstacles.crossing_ramps import CrossingRamps
from ftr_terrain_gen.obstacles.diagonal_ramp import DiagonalRamp
from ftr_terrain_gen.obstacles.diagonal_stairs import DiagonalStairs
from ftr_terrain_gen.obstacles.diagonal_pyramid import DiagonalPyramid
from ftr_terrain_gen.obstacles.diagonal_trunk import DiagonalTrunk
from ftr_terrain_gen.obstacles.double_trench import DoubleTrench
from ftr_terrain_gen.obstacles.flat_patch import FlatPatch
from ftr_terrain_gen.obstacles.half_platform import HalfPlatform
from ftr_terrain_gen.obstacles.log_crossing import LogCrossing
from ftr_terrain_gen.obstacles.lowered_platform import LoweredPlatform
from ftr_terrain_gen.obstacles.lowered_stairs import LoweredStairs
from ftr_terrain_gen.obstacles.offset_gate import OffsetGate
from ftr_terrain_gen.obstacles.pitch_ramp import PitchRamp
from ftr_terrain_gen.obstacles.raised_platform import RaisedPlatform
from ftr_terrain_gen.obstacles.raised_stairs import RaisedStairs
from ftr_terrain_gen.obstacles.rock_formation import RockFormation
from ftr_terrain_gen.obstacles.round_log import RoundLog
from ftr_terrain_gen.obstacles.sequence import Sequence
from ftr_terrain_gen.obstacles.stairs_ascent_descent import StairsAscentDescent
from ftr_terrain_gen.obstacles.steep_hill import BumpyHill, SteepHill, TwistedHill
from ftr_terrain_gen.obstacles.stepfield import Stepfield
from ftr_terrain_gen.obstacles.stump import Stump
from ftr_terrain_gen.obstacles.stump_field import StumpField
from ftr_terrain_gen.obstacles.tilted_pallet import TiltedPallet
from ftr_terrain_gen.obstacles.twin_rails import TwinRails
from ftr_terrain_gen.obstacles.wave import Wave
from ftr_terrain_gen.obstacles.widening_trench import WideningTrench

_CLASSES: list[type[Obstacle]] = [
    RaisedPlatform,
    DoubleTrench,
    WideningTrench,
    LogCrossing,
    TwinRails,
    DiagonalTrunk,
    RaisedStairs,
    LoweredStairs,
    Cobblestones,
    RockFormation,
    DiagonalPyramid,
    HalfPlatform,
    FlatPatch,
    Stump,
    StairsAscentDescent,
    # sloped / analytic surfaces (meshes) and the MARV-calibrated additions
    PitchRamp,
    DiagonalRamp,
    DiagonalStairs,
    CrossSlope,
    CrossingRamps,
    Wave,
    RoundLog,
    LoweredPlatform,
    Stepfield,
    StumpField,
    TiltedPallet,
    # steep hills: stability on a long steep slope (pitch only / changing roll / random bumps)
    SteepHill,
    TwistedHill,
    BumpyHill,
    OffsetGate,
    Sequence,
]

REGISTRY: dict[str, Obstacle] = {cls.name: cls() for cls in _CLASSES}


def get_obstacle(type_name: str) -> Obstacle:
    try:
        return REGISTRY[type_name]
    except KeyError:
        raise ValueError(
            f"Unknown obstacle type {type_name!r}. Known types: {sorted(REGISTRY)}"
        ) from None

from __future__ import annotations

from ftr_terrain_gen.obstacle_base import Obstacle
from ftr_terrain_gen.obstacles.cobblestones import Cobblestones
from ftr_terrain_gen.obstacles.diagonal_pyramid import DiagonalPyramid
from ftr_terrain_gen.obstacles.diagonal_trunk import DiagonalTrunk
from ftr_terrain_gen.obstacles.double_trench import DoubleTrench
from ftr_terrain_gen.obstacles.half_platform import HalfPlatform
from ftr_terrain_gen.obstacles.log_crossing import LogCrossing
from ftr_terrain_gen.obstacles.lowered_stairs import LoweredStairs
from ftr_terrain_gen.obstacles.raised_platform import RaisedPlatform
from ftr_terrain_gen.obstacles.raised_stairs import RaisedStairs
from ftr_terrain_gen.obstacles.rock_formation import RockFormation
from ftr_terrain_gen.obstacles.twin_rails import TwinRails
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
]

REGISTRY: dict[str, Obstacle] = {cls.name: cls() for cls in _CLASSES}


def get_obstacle(type_name: str) -> Obstacle:
    try:
        return REGISTRY[type_name]
    except KeyError:
        raise ValueError(
            f"Unknown obstacle type {type_name!r}. Known types: {sorted(REGISTRY)}"
        ) from None

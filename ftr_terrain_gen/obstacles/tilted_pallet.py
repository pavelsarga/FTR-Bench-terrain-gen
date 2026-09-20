from __future__ import annotations

import math

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, Obstacle, TileSpec
from ftr_terrain_gen.shapes import paint_tilted_rect
from ftr_terrain_gen.usd_utils import add_ground_slab, add_tilted_slab

PALLET_LENGTH = 1.2  # along the pallet's own (yawed) axis
PALLET_WIDTH = 1.6  # across — two euro pallets side by side, so the robot cannot go around
MIN_PITCH = 8.0  # tilt of the top face at the easiest repeat (degrees)
MAX_PITCH = 18.0
MIN_YAW = 10.0  # yaw of the pallet axis from the travel direction
MAX_YAW = 30.0
THICKNESS = 0.15


class TiltedPallet(Obstacle):
    """Číhala et al.'s "tilted pallet partially buried in the ground": a slab
    whose low edge is flush with the ground and whose top face rises
    `pitch` degrees along an axis turned `yaw` degrees from the travel
    direction, ending in a drop of length*sin(pitch) at the high edge. The
    robot climbs a ramp that is not square to it (roll + pitch), then falls
    off an edge that is not square either. Pitch and yaw both grade with
    `diff.t`; `extra.mirror_alternate` flips the yaw sign on odd repeats.
    Built with a single tilted box (native collider) over a ground slab;
    `diff.height` is ignored (use the pitch range).
    """

    name = "tilted_pallet"

    def _params(self, diff: DifficultyParams):
        length = diff.extra.get("pallet_length", PALLET_LENGTH)
        width = diff.extra.get("pallet_width", PALLET_WIDTH)
        pitch = diff.extra.get("min_pitch", MIN_PITCH) + (diff.extra.get("max_pitch", MAX_PITCH) - diff.extra.get("min_pitch", MIN_PITCH)) * diff.t
        yaw = diff.mirror_sign * (diff.extra.get("min_yaw", MIN_YAW) + (diff.extra.get("max_yaw", MAX_YAW) - diff.extra.get("min_yaw", MIN_YAW)) * diff.t)
        return length, width, pitch, yaw

    def _top_center(self, tile: TileSpec, length: float, pitch: float):
        # low edge flush with the ground: centre of the top face is half a length up the slope
        return (0.0, 0.0, tile.base_z + (length / 2) * math.sin(math.radians(pitch)))

    def build_usd(self, stage, prim_path: str, tile: TileSpec, diff: DifficultyParams) -> None:
        add_ground_slab(stage, f"{prim_path}/base", 0.0, 0.0, tile.width, tile.depth, tile.base_z, tile)
        length, width, pitch, yaw = self._params(diff)
        add_tilted_slab(
            stage, f"{prim_path}/pallet", self._top_center(tile, length, pitch), length, width, pitch, yaw,
            thickness=diff.extra.get("thickness", THICKNESS),
        )

    def build_heightmap(self, tile: TileSpec, diff: DifficultyParams) -> np.ndarray:
        h = self.flat_heightmap(tile)
        length, width, pitch, yaw = self._params(diff)
        return paint_tilted_rect(h, tile, self._top_center(tile, length, pitch), length, width, pitch, yaw)

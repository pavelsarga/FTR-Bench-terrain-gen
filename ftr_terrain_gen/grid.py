"""Course layout math: turns a list of row configs (obstacle type + height
range) into world-space tile placements, matching MapHelper's world-XY <->
array-index convention exactly (`terrain.py`'s `compensation` formula) so the
generated .map round-trips correctly with the consumer.

Every row has the same number of repeats (`CourseGrid.repeats`, course-wide)
so every row's tiles line up column-for-column with every other row's —
there is no filler/padding to reason about.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ftr_terrain_gen.obstacle_base import DifficultyParams, TileSpec


@dataclass(frozen=True)
class RowConfig:
    type: str
    min_height: float = 0.0
    max_height: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)
    # how `t` (0 = first repeat, 1 = last) maps to difficulty: "linear" (default),
    # "quadratic" (t^2: spends more repeats on the easy end) or "sqrt" (spreads the
    # repeats over the HARD end — the custom_mixed lesson: 4 of its 10 columns were
    # ~100% for every policy, the interesting range was the last 3)
    grade: str = "linear"
    # lateral offset of the TARGET from the lane centreline, [min, max] metres, graded
    # by t and alternating sign by repeat (see birth.py) — only for rows whose whole
    # tile is flat except the feature that makes the offset necessary (offset_gate)
    goal_lateral_offset: tuple[float, float] | None = None

    def graded_t(self, t: float) -> float:
        if self.grade == "quadratic":
            return t * t
        if self.grade == "sqrt":
            return t**0.5
        if self.grade != "linear":
            raise ValueError(f"unknown grade {self.grade!r} (linear|quadratic|sqrt)")
        return t


@dataclass(frozen=True)
class PlacedTile:
    row_index: int
    col_index: int
    obstacle_type: str
    x_center: float  # world X of tile center
    y_center: float  # world Y of tile center
    diff: DifficultyParams


class CourseGrid:
    def __init__(
        self,
        rows: list[RowConfig],
        repeats: int,
        tile_width: float,
        tile_depth: float,
        cell_size: float,
        base_z: float,
        border_width: float = 2.0,
        course_seed: int = 0,
    ) -> None:
        if repeats < 1:
            raise ValueError(f"repeats must be >= 1, got {repeats}")
        self.rows = rows
        self.repeats = repeats
        self.tile_width = tile_width
        self.tile_depth = tile_depth
        self.cell_size = cell_size
        self.base_z = base_z
        self.border_width = border_width
        self.course_seed = course_seed

        self.n_rows = len(rows)

    @property
    def course_width_x(self) -> float:
        return self.repeats * self.tile_width

    @property
    def course_depth_y(self) -> float:
        return self.n_rows * self.tile_depth

    @property
    def map_lower(self) -> tuple[float, float, float]:
        return (
            -(self.course_width_x / 2 + self.border_width),
            -(self.course_depth_y / 2 + self.border_width),
            -1.0,
        )

    @property
    def map_upper(self) -> tuple[float, float, float]:
        return (
            self.course_width_x / 2 + self.border_width,
            self.course_depth_y / 2 + self.border_width,
            3.0,
        )

    @property
    def compensation(self) -> np.ndarray:
        lower = np.array(self.map_lower[:2])
        return -(lower / self.cell_size).astype(np.int32)

    @property
    def map_shape(self) -> tuple[int, int]:
        lower = np.array(self.map_lower[:2])
        upper = np.array(self.map_upper[:2])
        return tuple(np.round((upper - lower) / self.cell_size).astype(int))

    def world_to_index(self, world_xy: tuple[float, float]) -> tuple[int, int]:
        idx = np.floor(np.array(world_xy) / self.cell_size + self.compensation).astype(int)
        return int(idx[0]), int(idx[1])

    def tile_spec(self) -> TileSpec:
        return TileSpec(
            width=self.tile_width, depth=self.tile_depth, cell_size=self.cell_size, base_z=self.base_z
        )

    def _grid_x_center(self, col_index: int) -> float:
        x0 = -self.course_width_x / 2 + self.tile_width / 2
        return x0 + col_index * self.tile_width

    def _seed_for(self, row_index: int, col_index: int) -> int:
        seq = np.random.SeedSequence([self.course_seed, row_index, col_index])
        return int(seq.generate_state(1)[0])

    def iter_tiles(self):
        """Yield a PlacedTile for every (row, col) slot in the course."""
        for row_index, row in enumerate(self.rows):
            y_center = (row_index - (self.n_rows - 1) / 2) * self.tile_depth
            for col_index in range(self.repeats):
                t = 0.0 if self.repeats <= 1 else row.graded_t(col_index / (self.repeats - 1))
                height = row.min_height + (row.max_height - row.min_height) * t
                seed = self._seed_for(row_index, col_index)
                diff = DifficultyParams(t=t, height=height, seed=seed, extra=row.extra, col_index=col_index)
                yield PlacedTile(
                    row_index=row_index,
                    col_index=col_index,
                    obstacle_type=row.type,
                    x_center=self._grid_x_center(col_index),
                    y_center=y_center,
                    diff=diff,
                )

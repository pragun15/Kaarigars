"""Procedural map generation."""

from __future__ import annotations

import random

from rescue_env.models.state import MapData


class MapGenerator:
    def generate(self, map_size: int, debris_density: float, rng: random.Random) -> MapData:
        occupancy: list[list[int]] = []
        debris: list[list[float]] = []

        for _ in range(map_size):
            occ_row: list[int] = []
            debris_row: list[float] = []
            for _ in range(map_size):
                has_obstacle = 1 if rng.random() < debris_density * 0.45 else 0
                occ_row.append(has_obstacle)
                debris_row.append(rng.uniform(0.0, 1.0) if has_obstacle else 0.0)
            occupancy.append(occ_row)
            debris.append(debris_row)

        # Keep start area always clear.
        for y in range(min(3, map_size)):
            for x in range(min(3, map_size)):
                occupancy[y][x] = 0
                debris[y][x] = 0.0

        return MapData(width=map_size, height=map_size, occupancy=occupancy, debris=debris)

"""Debris modifier utilities."""

from __future__ import annotations

import random

from rescue_env.models.state import MapData


class DebrisSystem:
    def enrich(self, map_data: MapData, rng: random.Random, extra_density: float) -> MapData:
        for y in range(map_data.height):
            for x in range(map_data.width):
                if map_data.occupancy[y][x] == 1:
                    map_data.debris[y][x] = min(1.0, map_data.debris[y][x] + rng.uniform(0.0, extra_density))
                elif rng.random() < extra_density * 0.05:
                    map_data.occupancy[y][x] = 1
                    map_data.debris[y][x] = rng.uniform(0.1, 0.6)
        return map_data

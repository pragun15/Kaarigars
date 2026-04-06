"""LiDAR sensor hook."""

from __future__ import annotations

import random

from rescue_env.models.robot import Position


class LidarSensor:
    def scan(self, position: Position, noise: float, rng: random.Random) -> list[tuple[float, float, float, float]]:
        points: list[tuple[float, float, float, float]] = []
        for _ in range(16):
            dx = rng.uniform(-5.0, 5.0)
            dy = rng.uniform(-5.0, 5.0)
            dz = rng.uniform(-0.5, 2.0)
            n = rng.uniform(-noise, noise)
            points.append((position.x + dx + n, position.y + dy + n, max(0.0, dz), rng.uniform(0.2, 1.0)))
        return points

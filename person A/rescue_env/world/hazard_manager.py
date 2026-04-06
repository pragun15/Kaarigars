"""Hazard generation and checks."""

from __future__ import annotations

import math
import random

from rescue_env.models.state import HazardZone


class HazardManager:
    def generate(self, map_size: int, hazard_density: float, rng: random.Random) -> list[HazardZone]:
        hazard_count = max(1, int(map_size * map_size * hazard_density * 0.03))
        hazards: list[HazardZone] = []
        for idx in range(hazard_count):
            hazards.append(
                HazardZone(
                    hazard_id=f"hazard_{idx}",
                    hazard_type="gas",
                    x=rng.randint(0, map_size - 1),
                    y=rng.randint(0, map_size - 1),
                    radius=rng.uniform(1.5, 3.5),
                    severity=rng.uniform(0.2, 1.0),
                )
            )
        return hazards

    def in_critical_gas_zone(self, x: float, y: float, hazards: list[HazardZone]) -> bool:
        for hazard in hazards:
            if hazard.hazard_type != "gas":
                continue
            distance = math.sqrt((x - hazard.x) ** 2 + (y - hazard.y) ** 2)
            if distance <= hazard.radius and hazard.severity >= 0.7:
                return True
        return False

"""Gas sensor hook."""

from __future__ import annotations

import random

from rescue_env.models.observations import GasReading
from rescue_env.models.robot import Position
from rescue_env.models.state import HazardZone


class GasSensor:
    def measure(self, position: Position, hazards: list[HazardZone], noise: float, rng: random.Random) -> GasReading:
        co = 5.0
        ch4 = 1.0
        co2 = 420.0
        h2s = 0.3
        o2 = 20.9

        for hazard in hazards:
            distance = ((position.x - hazard.x) ** 2 + (position.y - hazard.y) ** 2) ** 0.5
            if distance <= hazard.radius:
                factor = max(0.1, 1.0 - (distance / max(hazard.radius, 0.1))) * hazard.severity
                co += 80.0 * factor
                ch4 += 15.0 * factor
                co2 += 600.0 * factor
                h2s += 10.0 * factor
                o2 -= 3.0 * factor

        def with_noise(v: float) -> float:
            return max(0.0, v + rng.uniform(-noise, noise) * 10.0)

        return GasReading(o2=with_noise(o2), co=with_noise(co), ch4=with_noise(ch4), co2=with_noise(co2), h2s=with_noise(h2s))

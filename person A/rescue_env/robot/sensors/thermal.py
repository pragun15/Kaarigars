"""Thermal sensor hook."""

from __future__ import annotations

import random

from rescue_env.models.observations import ThermalSignature
from rescue_env.models.robot import Position
from rescue_env.models.victim import Victim


class ThermalSensor:
    def scan(self, victims: list[Victim], robot_pos: Position, noise: float, rng: random.Random) -> list[ThermalSignature]:
        signatures: list[ThermalSignature] = []
        for victim in victims:
            if victim.rescued:
                continue
            distance = ((victim.position.x - robot_pos.x) ** 2 + (victim.position.y - robot_pos.y) ** 2) ** 0.5
            if distance <= 8.0:
                confidence = max(0.1, 1.0 - distance / 10.0 - noise * 0.5 + rng.uniform(-0.1, 0.1))
                signatures.append(
                    ThermalSignature(position=victim.position, temperature_c=36.5 + rng.uniform(-1.2, 1.2), confidence=confidence)
                )
        return signatures

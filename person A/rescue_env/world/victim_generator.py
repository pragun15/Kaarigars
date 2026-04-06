"""Victim creation using difficulty-aligned distributions."""

from __future__ import annotations

import random

from rescue_env.models.robot import Position
from rescue_env.models.victim import EntrapmentStatus, HealthStatus, Victim


class VictimGenerator:
    def generate(self, victim_range: tuple[int, int], map_size: int, rng: random.Random) -> list[Victim]:
        total = rng.randint(victim_range[0], victim_range[1])
        victims: list[Victim] = []

        for idx in range(total):
            health_roll = rng.random()
            if health_roll < 0.15:
                health = HealthStatus.CRITICAL
            elif health_roll < 0.45:
                health = HealthStatus.VULNERABLE
            else:
                health = HealthStatus.HEALTHY

            entrapment_roll = rng.random()
            if entrapment_roll < 0.2:
                entrapment = EntrapmentStatus.TRAPPED
            elif entrapment_roll < 0.5:
                entrapment = EntrapmentStatus.PINNED
            else:
                entrapment = EntrapmentStatus.FREE

            priority = self._priority(health, entrapment)
            victims.append(
                Victim(
                    victim_id=f"victim_{idx}",
                    position=Position(x=rng.uniform(0, map_size - 1), y=rng.uniform(0, map_size - 1), z=0.0),
                    health=health,
                    entrapment=entrapment,
                    priority_score=priority,
                )
            )

        return victims

    def _priority(self, health: HealthStatus, entrapment: EntrapmentStatus) -> float:
        health_score = {HealthStatus.CRITICAL: 1.0, HealthStatus.VULNERABLE: 0.6, HealthStatus.HEALTHY: 0.2}[health]
        entrapment_score = {EntrapmentStatus.TRAPPED: 1.0, EntrapmentStatus.PINNED: 0.6, EntrapmentStatus.FREE: 0.2}[entrapment]
        return round(0.7 * health_score + 0.3 * entrapment_score, 3)

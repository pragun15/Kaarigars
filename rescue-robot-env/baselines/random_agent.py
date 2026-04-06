from __future__ import annotations

import random
from typing import Any


class RandomAgent:
    """Simple random policy over the action set."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.action_types = [
            "move",
            "scan_lidar",
            "scan_thermal",
            "scan_gas",
            "listen",
            "rescue_victim",
            "flag_hazard",
            "idle",
        ]

    def act(self, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        action_type = self.rng.choice(self.action_types)

        if action_type == "move":
            return {
                "action_type": "move",
                "parameters": {
                    "target_position": [self.rng.uniform(-15, 15), self.rng.uniform(-15, 15)],
                    "speed": self.rng.uniform(0.2, 1.6),
                },
            }

        if action_type == "rescue_victim":
            return {
                "action_type": "rescue_victim",
                "parameters": {
                    "victim_id": "unknown",
                    "handling_method": self.rng.choice(["gentle", "standard", "emergency"]),
                },
            }

        if action_type == "flag_hazard":
            return {
                "action_type": "flag_hazard",
                "parameters": {
                    "hazard_type": self.rng.choice(["gas", "structural", "crush_risk"]),
                    "location": [self.rng.uniform(-10, 10), self.rng.uniform(-10, 10)],
                },
            }

        return {
            "action_type": action_type,
            "parameters": {},
        }

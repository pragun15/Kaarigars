"""Acoustic sensor hook."""

from __future__ import annotations

import math
import random

from rescue_env.models.observations import AcousticEvent
from rescue_env.models.robot import Position
from rescue_env.models.victim import Victim


class AcousticSensor:
    def listen(self, victims: list[Victim], robot_pos: Position, noise: float, rng: random.Random) -> list[AcousticEvent]:
        events: list[AcousticEvent] = []
        for victim in victims:
            if victim.rescued:
                continue
            distance = ((victim.position.x - robot_pos.x) ** 2 + (victim.position.y - robot_pos.y) ** 2) ** 0.5
            if distance <= 12.0 and rng.random() > noise:
                direction = math.degrees(math.atan2(victim.position.y - robot_pos.y, victim.position.x - robot_pos.x))
                confidence = max(0.1, 1.0 - distance / 15.0 - noise * 0.4)
                events.append(AcousticEvent(event_type="human_voice", direction_deg=direction, confidence=confidence))
        return events

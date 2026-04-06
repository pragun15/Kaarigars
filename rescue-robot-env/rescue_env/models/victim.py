"""Victim related models."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel

from rescue_env.models.robot import Position


class HealthStatus(str, Enum):
    HEALTHY = "healthy"
    VULNERABLE = "vulnerable"
    CRITICAL = "critical"


class EntrapmentStatus(str, Enum):
    FREE = "free"
    PINNED = "pinned"
    TRAPPED = "trapped"


class Victim(BaseModel):
    victim_id: str
    position: Position
    health: HealthStatus
    entrapment: EntrapmentStatus
    detected: bool = False
    rescued: bool = False
    priority_score: float = 0.0

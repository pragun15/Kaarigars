"""Robot models and runtime state."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Position(BaseModel):
    x: float
    y: float
    z: float = 0.0


class RobotSpecs(BaseModel):
    max_speed_ms: float = Field(default=1.6, ge=0.8, le=1.6)
    weight_kg: float = Field(default=26.0, ge=20.0, le=32.0)
    endurance_minutes: float = Field(default=100.0, ge=90.0, le=120.0)


class RobotState(BaseModel):
    position: Position
    orientation_deg: float = 0.0
    battery_level: float = Field(default=100.0, ge=0.0, le=100.0)
    is_stable: bool = True
    is_operational: bool = True
    carrying_victim_id: str | None = None
    collisions: int = 0
    instability_events: int = 0
    successful_recoveries: int = 0

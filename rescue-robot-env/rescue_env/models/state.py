"""Serialized environment state model."""

from __future__ import annotations

from pydantic import BaseModel

from rescue_env.models.robot import RobotState
from rescue_env.models.victim import Victim


class HazardZone(BaseModel):
    hazard_id: str
    hazard_type: str
    x: int
    y: int
    radius: float
    severity: float


class MapData(BaseModel):
    width: int
    height: int
    occupancy: list[list[int]]
    debris: list[list[float]]


class MissionMetrics(BaseModel):
    steps_taken: int = 0
    time_elapsed: float = 0.0
    victims_detected: int = 0
    victims_rescued: int = 0
    collisions: int = 0


class StateSnapshot(BaseModel):
    episode_id: str
    difficulty: str
    seed: int
    map_data: MapData
    victims: list[Victim]
    hazards: list[HazardZone]
    robot: RobotState
    metrics: MissionMetrics

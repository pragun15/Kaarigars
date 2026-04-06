"""Observation models returned by reset and step."""

from __future__ import annotations

from typing import List

from pydantic import BaseModel

from rescue_env.models.robot import Position
from rescue_env.models.victim import HealthStatus


class ThermalSignature(BaseModel):
    position: Position
    temperature_c: float
    confidence: float


class GasReading(BaseModel):
    o2: float
    co: float
    ch4: float
    co2: float
    h2s: float


class AcousticEvent(BaseModel):
    event_type: str
    direction_deg: float
    confidence: float


class VictimObservation(BaseModel):
    victim_id: str
    position: Position
    health: HealthStatus
    confidence: float


class HazardObservation(BaseModel):
    hazard_type: str
    position: Position
    severity: float


class RobotStatusObservation(BaseModel):
    position: Position
    orientation_deg: float
    battery_level: float
    is_stable: bool
    carrying_victim: bool


class SensorReadings(BaseModel):
    lidar_points: List[tuple[float, float, float, float]]
    thermal_signatures: List[ThermalSignature]
    gas_levels: GasReading
    acoustic_events: List[AcousticEvent]


class Observation(BaseModel):
    robot_status: RobotStatusObservation
    sensors: SensorReadings
    nearby_victims: List[VictimObservation]
    nearby_hazards: List[HazardObservation]
    time_remaining: float
    mission_progress: float

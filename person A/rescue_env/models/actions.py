"""Action space models."""

from __future__ import annotations

from enum import Enum
from typing import Optional, Tuple, Union

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    MOVE = "move"
    SCAN_LIDAR = "scan_lidar"
    SCAN_THERMAL = "scan_thermal"
    SCAN_GAS = "scan_gas"
    LISTEN = "listen"
    RESCUE_VICTIM = "rescue_victim"
    FLAG_HAZARD = "flag_hazard"
    IDLE = "idle"


class MoveAction(BaseModel):
    type: ActionType = ActionType.MOVE
    target_position: Tuple[float, float]
    speed: float = Field(default=0.8, ge=0.0, le=1.6)


class ScanAction(BaseModel):
    type: ActionType
    direction: Optional[float] = None
    duration: float = Field(default=1.0, ge=0.1, le=10.0)


class RescueAction(BaseModel):
    type: ActionType = ActionType.RESCUE_VICTIM
    victim_id: str
    handling_method: str = Field(default="gentle")


class FlagHazardAction(BaseModel):
    type: ActionType = ActionType.FLAG_HAZARD
    hazard_type: str
    location: Tuple[float, float]


class IdleAction(BaseModel):
    type: ActionType = ActionType.IDLE
    duration: float = Field(default=1.0, ge=0.1, le=30.0)


Action = Union[MoveAction, ScanAction, RescueAction, FlagHazardAction, IdleAction]

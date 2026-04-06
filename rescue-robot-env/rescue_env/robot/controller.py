"""Action execution for robot behavior."""

from __future__ import annotations

from dataclasses import dataclass

from rescue_env.models.actions import Action, ActionType, MoveAction, RescueAction
from rescue_env.models.robot import RobotState
from rescue_env.models.state import HazardZone, MapData
from rescue_env.models.victim import Victim
from rescue_env.robot.battery import BatteryModel
from rescue_env.world.physics_engine import PhysicsEngine


@dataclass
class ActionResult:
    action_type: ActionType
    time_cost_minutes: float
    battery_cost: float
    moved_distance: float = 0.0
    collision: bool = False
    rescued_victim_id: str | None = None


class RobotController:
    def __init__(self, physics_engine: PhysicsEngine, battery_model: BatteryModel) -> None:
        self.physics_engine = physics_engine
        self.battery_model = battery_model

    def execute(
        self,
        action: Action,
        robot: RobotState,
        map_data: MapData,
        victims: list[Victim],
        hazards: list[HazardZone],
    ) -> ActionResult:
        _ = hazards
        if action.type == ActionType.MOVE and isinstance(action, MoveAction):
            movement = self.physics_engine.move(robot, action.target_position, action.speed, map_data)
            cost = self.battery_model.consume(ActionType.MOVE, movement.time_cost_minutes, movement.moved_distance)
            return ActionResult(
                action_type=ActionType.MOVE,
                time_cost_minutes=max(movement.time_cost_minutes, 0.05),
                battery_cost=cost,
                moved_distance=movement.moved_distance,
                collision=movement.collision,
            )

        if action.type == ActionType.RESCUE_VICTIM and isinstance(action, RescueAction):
            rescued = self._rescue(action.victim_id, robot, victims)
            time_cost = 0.4
            cost = self.battery_model.consume(ActionType.RESCUE_VICTIM, time_cost)
            return ActionResult(
                action_type=ActionType.RESCUE_VICTIM,
                time_cost_minutes=time_cost,
                battery_cost=cost,
                rescued_victim_id=rescued,
            )

        time_cost = 0.15
        battery_cost = self.battery_model.consume(action.type, time_cost)
        return ActionResult(action_type=action.type, time_cost_minutes=time_cost, battery_cost=battery_cost)

    def _rescue(self, victim_id: str, robot: RobotState, victims: list[Victim]) -> str | None:
        for victim in victims:
            if victim.victim_id != victim_id or victim.rescued:
                continue
            distance = ((victim.position.x - robot.position.x) ** 2 + (victim.position.y - robot.position.y) ** 2) ** 0.5
            if distance <= 2.0:
                victim.rescued = True
                return victim_id
        return None

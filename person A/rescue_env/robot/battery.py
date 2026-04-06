"""Battery drain model based on difficulty profile."""

from __future__ import annotations

from rescue_env.models.actions import ActionType


class BatteryModel:
    def __init__(self, move_rate: float, sensor_rate: float, idle_rate: float) -> None:
        self.move_rate = move_rate
        self.sensor_rate = sensor_rate
        self.idle_rate = idle_rate

    def consume(
        self,
        action_type: ActionType,
        duration_minutes: float,
        moved_distance: float = 0.0,
    ) -> float:
        duration = max(duration_minutes, 0.0)
        if action_type == ActionType.MOVE:
            return duration * self.move_rate + moved_distance * 0.02
        if action_type in {
            ActionType.SCAN_LIDAR,
            ActionType.SCAN_THERMAL,
            ActionType.SCAN_GAS,
            ActionType.LISTEN,
        }:
            return duration * self.sensor_rate
        if action_type == ActionType.IDLE:
            return duration * self.idle_rate
        return duration * (self.idle_rate + 0.05)

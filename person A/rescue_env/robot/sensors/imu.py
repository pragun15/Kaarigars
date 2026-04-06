"""IMU sensor hook."""

from __future__ import annotations

from rescue_env.models.robot import RobotState


class ImuSensor:
    def read(self, robot: RobotState) -> dict[str, float]:
        return {
            "orientation_deg": robot.orientation_deg,
            "stability": 1.0 if robot.is_stable else 0.0,
        }

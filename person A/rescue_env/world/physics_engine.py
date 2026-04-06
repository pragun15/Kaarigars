"""Basic movement and collision simulation."""

from __future__ import annotations

from dataclasses import dataclass

from rescue_env.models.robot import Position, RobotState
from rescue_env.models.state import MapData


@dataclass
class MovementResult:
    moved_distance: float
    collision: bool
    instability_event: bool
    recovered: bool
    time_cost_minutes: float


class PhysicsEngine:
    def move(self, robot: RobotState, target: tuple[float, float], speed: float, map_data: MapData) -> MovementResult:
        clamped_x = min(max(target[0], 0.0), float(map_data.width - 1))
        clamped_y = min(max(target[1], 0.0), float(map_data.height - 1))

        grid_x = int(round(clamped_x))
        grid_y = int(round(clamped_y))

        old_x, old_y = robot.position.x, robot.position.y
        moved_distance = ((clamped_x - old_x) ** 2 + (clamped_y - old_y) ** 2) ** 0.5

        collision = map_data.occupancy[grid_y][grid_x] == 1
        instability = False
        recovered = False

        if collision:
            robot.collisions += 1
            moved_distance = 0.0
        else:
            robot.position = Position(x=clamped_x, y=clamped_y, z=robot.position.z)

        # Simple stability proxy from debris around target.
        local_debris = map_data.debris[grid_y][grid_x]
        if local_debris > 0.75:
            robot.instability_events += 1
            instability = True
            if local_debris < 0.9:
                robot.successful_recoveries += 1
                recovered = True

        ms = max(speed, 0.1)
        time_cost_minutes = (moved_distance / ms) / 60.0
        return MovementResult(
            moved_distance=moved_distance,
            collision=collision,
            instability_event=instability,
            recovered=recovered,
            time_cost_minutes=time_cost_minutes,
        )

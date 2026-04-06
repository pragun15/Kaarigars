"""Main OpenEnv-compatible rescue environment shell."""

from __future__ import annotations

import random
from typing import Any
from uuid import uuid4

from rescue_env.core.config import EnvConfig, get_profile
from rescue_env.models.actions import Action, ActionType
from rescue_env.models.observations import (
    GasReading,
    HazardObservation,
    Observation,
    RobotStatusObservation,
    SensorReadings,
    VictimObservation,
)
from rescue_env.models.robot import Position, RobotSpecs, RobotState
from rescue_env.models.state import HazardZone, MapData, MissionMetrics, StateSnapshot
from rescue_env.models.victim import Victim
from rescue_env.robot.battery import BatteryModel
from rescue_env.robot.controller import RobotController
from rescue_env.robot.sensors.acoustic import AcousticSensor
from rescue_env.robot.sensors.gas_sensor import GasSensor
from rescue_env.robot.sensors.imu import ImuSensor
from rescue_env.robot.sensors.lidar import LidarSensor
from rescue_env.robot.sensors.thermal import ThermalSensor
from rescue_env.world.debris_system import DebrisSystem
from rescue_env.world.hazard_manager import HazardManager
from rescue_env.world.map_generator import MapGenerator
from rescue_env.world.physics_engine import PhysicsEngine
from rescue_env.world.victim_generator import VictimGenerator


class RescueEnvironment:
    def __init__(self, difficulty: str = "easy", config: dict[str, Any] | None = None):
        cfg_input = {"difficulty": difficulty}
        if config:
            cfg_input.update(config)
        self.config = EnvConfig(**cfg_input)
        self.profile = get_profile(self.config.difficulty)
        self.specs = RobotSpecs()

        self._base_seed = self.config.seed
        self._episode_index = 0
        self._rng = random.Random(self._base_seed)
        self._episode_id = ""

        self.map_data: MapData | None = None
        self.hazards: list[HazardZone] = []
        self.victims: list[Victim] = []
        self.robot: RobotState | None = None
        self.metrics = MissionMetrics()

        self.map_generator = MapGenerator()
        self.debris_system = DebrisSystem()
        self.hazard_manager = HazardManager()
        self.victim_generator = VictimGenerator()

        battery_profile = self.profile["battery"]
        self.physics_engine = PhysicsEngine()
        self.battery_model = BatteryModel(
            move_rate=battery_profile["move"],
            sensor_rate=battery_profile["sensor"],
            idle_rate=battery_profile["idle"],
        )
        self.controller = RobotController(self.physics_engine, self.battery_model)

        self.lidar = LidarSensor()
        self.thermal = ThermalSensor()
        self.gas_sensor = GasSensor()
        self.acoustic = AcousticSensor()
        self.imu = ImuSensor()

    def reset(self, seed: int | None = None) -> Observation:
        if seed is None:
            seed = self._base_seed + self._episode_index
        self._episode_index += 1
        self._rng = random.Random(seed)
        self._episode_id = str(uuid4())

        map_size = self.profile["map_size"]
        self.map_data = self.map_generator.generate(map_size, self.profile["debris_density"], self._rng)
        self.map_data = self.debris_system.enrich(self.map_data, self._rng, extra_density=self.profile["debris_density"])
        self.hazards = self.hazard_manager.generate(map_size, self.profile["hazard_density"], self._rng)
        self.victims = self.victim_generator.generate(self.profile["victim_range"], map_size, self._rng)

        self.robot = RobotState(position=Position(x=0.0, y=0.0, z=0.0), battery_level=self.profile["battery_capacity"])
        self.metrics = MissionMetrics()
        return self._build_observation()

    def step(self, action: Action) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._require_initialized()

        assert self.robot is not None
        assert self.map_data is not None

        prev_detected = self.metrics.victims_detected
        prev_rescued = self.metrics.victims_rescued

        result = self.controller.execute(action, self.robot, self.map_data, self.victims, self.hazards)

        self.robot.battery_level = max(0.0, self.robot.battery_level - result.battery_cost)
        self.metrics.steps_taken += 1
        self.metrics.time_elapsed += result.time_cost_minutes
        if result.collision:
            self.metrics.collisions += 1

        self._update_detection_from_passive_sensors()
        self.metrics.victims_rescued = len([v for v in self.victims if v.rescued])

        reward = self._step_reward(action.type, prev_detected, prev_rescued, result.collision, result.rescued_victim_id is not None)
        done, truncated, reason = self._termination()

        observation = self._build_observation()
        info = {
            "episode_id": self._episode_id,
            "difficulty": self.config.difficulty,
            "metrics": self.metrics.model_dump(),
            "reason": reason,
            "success": done and self.metrics.victims_rescued == len(self.victims),
        }
        return observation, reward, done, truncated, info

    def state(self) -> StateSnapshot:
        self._require_initialized()
        assert self.map_data is not None
        assert self.robot is not None
        return StateSnapshot(
            episode_id=self._episode_id,
            difficulty=self.config.difficulty,
            seed=self._base_seed,
            map_data=self.map_data,
            victims=self.victims,
            hazards=self.hazards,
            robot=self.robot,
            metrics=self.metrics,
        )

    def _build_observation(self) -> Observation:
        self._require_initialized()
        assert self.robot is not None

        noise = self.profile["sensor_noise"]
        lidar_points = self.lidar.scan(self.robot.position, noise, self._rng)
        thermal_signatures = self.thermal.scan(self.victims, self.robot.position, noise, self._rng)
        gas_levels = self.gas_sensor.measure(self.robot.position, self.hazards, noise, self._rng)
        acoustic_events = self.acoustic.listen(self.victims, self.robot.position, noise, self._rng)
        _ = self.imu.read(self.robot)

        nearby_victims = self._nearby_victim_observations()
        nearby_hazards = self._nearby_hazard_observations()
        time_remaining = max(0.0, self.profile["time_limit_minutes"] - self.metrics.time_elapsed)
        progress = self.metrics.victims_rescued / max(1, len(self.victims))

        return Observation(
            robot_status=RobotStatusObservation(
                position=self.robot.position,
                orientation_deg=self.robot.orientation_deg,
                battery_level=self.robot.battery_level,
                is_stable=self.robot.is_stable,
                carrying_victim=self.robot.carrying_victim_id is not None,
            ),
            sensors=SensorReadings(
                lidar_points=lidar_points,
                thermal_signatures=thermal_signatures,
                gas_levels=gas_levels,
                acoustic_events=acoustic_events,
            ),
            nearby_victims=nearby_victims,
            nearby_hazards=nearby_hazards,
            time_remaining=time_remaining,
            mission_progress=progress,
        )

    def _nearby_victim_observations(self) -> list[VictimObservation]:
        assert self.robot is not None
        obs: list[VictimObservation] = []
        for victim in self.victims:
            if victim.rescued:
                continue
            dist = ((victim.position.x - self.robot.position.x) ** 2 + (victim.position.y - self.robot.position.y) ** 2) ** 0.5
            if dist <= 10.0:
                confidence = max(0.1, 1.0 - dist / 12.0)
                obs.append(
                    VictimObservation(
                        victim_id=victim.victim_id,
                        position=victim.position,
                        health=victim.health,
                        confidence=confidence,
                    )
                )
                victim.detected = True
        self.metrics.victims_detected = len([v for v in self.victims if v.detected])
        return obs

    def _nearby_hazard_observations(self) -> list[HazardObservation]:
        assert self.robot is not None
        obs: list[HazardObservation] = []
        for hazard in self.hazards:
            dist = ((hazard.x - self.robot.position.x) ** 2 + (hazard.y - self.robot.position.y) ** 2) ** 0.5
            if dist <= 8.0:
                obs.append(
                    HazardObservation(
                        hazard_type=hazard.hazard_type,
                        position=Position(x=float(hazard.x), y=float(hazard.y), z=0.0),
                        severity=hazard.severity,
                    )
                )
        return obs

    def _update_detection_from_passive_sensors(self) -> None:
        assert self.robot is not None
        for victim in self.victims:
            if victim.rescued:
                continue
            dist = ((victim.position.x - self.robot.position.x) ** 2 + (victim.position.y - self.robot.position.y) ** 2) ** 0.5
            if dist <= 7.5:
                victim.detected = True
        self.metrics.victims_detected = len([v for v in self.victims if v.detected])

    def _step_reward(
        self,
        action_type: ActionType,
        prev_detected: int,
        prev_rescued: int,
        had_collision: bool,
        rescued_event: bool,
    ) -> float:
        reward = 0.0
        new_detected = max(0, self.metrics.victims_detected - prev_detected)
        new_rescued = max(0, self.metrics.victims_rescued - prev_rescued)

        reward += new_detected * 0.05
        reward += new_rescued * 0.2
        if rescued_event:
            reward += 0.1
        if action_type == ActionType.FLAG_HAZARD:
            reward += 0.03
        if had_collision:
            reward -= 0.08

        if self.hazard_manager.in_critical_gas_zone(self.robot.position.x, self.robot.position.y, self.hazards):
            reward -= 0.15

        battery_efficiency = self.robot.battery_level / 100.0
        reward += 0.01 * battery_efficiency
        return max(0.0, min(1.0, reward))

    def _termination(self) -> tuple[bool, bool, str | None]:
        assert self.robot is not None
        if not self.robot.is_operational:
            return True, False, "robot_destroyed"

        if self.metrics.victims_rescued == len(self.victims) and len(self.victims) > 0:
            return True, False, "all_victims_rescued"

        if self.robot.battery_level <= 0.0:
            return False, True, "battery_depleted"

        if self.metrics.time_elapsed >= self.profile["time_limit_minutes"]:
            return False, True, "time_limit_reached"

        if self.metrics.steps_taken >= self.config.max_steps:
            return False, True, "max_steps_reached"

        return False, False, None

    def _require_initialized(self) -> None:
        if self.robot is None or self.map_data is None:
            raise RuntimeError("Environment is not initialized. Call reset() first.")

"""Main OpenEnv-compatible rescue environment shell."""

from __future__ import annotations

import math
import random
from typing import Any
from uuid import uuid4

from rescue_env.core.config import EnvConfig, get_profile
from rescue_env.models.actions import (
    Action,
    ActionType,
    FlagHazardAction,
    IdleAction,
    MoveAction,
    RescueAction,
    ScanAction,
)
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
from rescue_env.models.victim import HealthStatus, Victim
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
from rescue_env.scoring.grader import grade_episode_by_difficulty
from rescue_env.scoring.reward_calculator import calculate_final_reward, calculate_step_reward
from rescue_env.scoring.types import EpisodeStats


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

        self._last_episode_stats: EpisodeStats | None = None
        self._visited_cells: set[tuple[int, int]] = set()
        self._revisit_steps = 0
        self._idle_steps = 0
        self._scan_actions = 0
        self._useful_insights = 0
        self._redundant_scans = 0
        self._hazards_flagged = 0
        self._rescue_attempts = 0
        self._successful_rescues = 0
        self._correct_priority_rescues = 0
        self._incorrect_rescue_order_events = 0
        self._false_positive_detections = 0
        self._detection_confidence_samples: list[float] = []
        self._critical_gas_zone_entries = 0
        self._preventable_destruction_events = 0
        self._collisions_near_survivor = 0
        self._last_action_result = "episode_reset"
        self._last_action_rejected: str | None = None
        self._last_hint = "Sweep nearby zones and use thermal/lidar to locate victims."
        self._no_progress_steps = 0
        self._last_move_target: tuple[float, float] | None = None
        self._same_move_target_streak = 0

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

        self._visited_cells = set()
        self._revisit_steps = 0
        self._idle_steps = 0
        self._scan_actions = 0
        self._useful_insights = 0
        self._redundant_scans = 0
        self._hazards_flagged = 0
        self._rescue_attempts = 0
        self._successful_rescues = 0
        self._correct_priority_rescues = 0
        self._incorrect_rescue_order_events = 0
        self._false_positive_detections = 0
        self._detection_confidence_samples = []
        self._critical_gas_zone_entries = 0
        self._preventable_destruction_events = 0
        self._collisions_near_survivor = 0
        self._last_action_result = "episode_reset"
        self._last_action_rejected = None
        self._last_hint = "Sweep nearby zones and use thermal/lidar to locate victims."
        self._no_progress_steps = 0
        self._last_move_target = None
        self._same_move_target_streak = 0

        self._mark_cell_visit(self.robot.position)
        self._last_episode_stats = self._build_episode_stats()
        return self._build_observation()

    def step(self, action: Action | dict[str, Any]) -> tuple[Observation, float, bool, bool, dict[str, Any]]:
        self._require_initialized()

        assert self.robot is not None
        assert self.map_data is not None

        parsed_action = self._normalize_action(action)
        prev_detected = self.metrics.victims_detected
        prev_rescued = self.metrics.victims_rescued
        prev_visited = len(self._visited_cells)
        prev_nearest_victim_distance = self._nearest_unrescued_victim_distance()
        previous_stats = self._last_episode_stats or self._build_episode_stats()

        parsed_action, rejection_reason, action_result = self._apply_action_guardrails(parsed_action)
        self._last_action_rejected = rejection_reason
        self._last_action_result = action_result

        had_critical_before = any(v.health == HealthStatus.CRITICAL and not v.rescued for v in self.victims)
        if parsed_action.type == ActionType.RESCUE_VICTIM:
            self._rescue_attempts += 1
        if parsed_action.type in {ActionType.SCAN_LIDAR, ActionType.SCAN_THERMAL, ActionType.SCAN_GAS, ActionType.LISTEN}:
            self._scan_actions += 1
        if parsed_action.type == ActionType.IDLE:
            self._idle_steps += 1
        if parsed_action.type == ActionType.FLAG_HAZARD:
            self._hazards_flagged += 1

        result = self.controller.execute(parsed_action, self.robot, self.map_data, self.victims, self.hazards)

        self.robot.battery_level = max(0.0, self.robot.battery_level - result.battery_cost)
        self.metrics.steps_taken += 1
        self.metrics.time_elapsed += result.time_cost_minutes
        if result.collision:
            self.metrics.collisions += 1
            if self._is_near_any_unrescued_victim(3.0):
                self._collisions_near_survivor += 1

        self._mark_cell_visit(self.robot.position)

        self._update_detection_from_passive_sensors()
        self.metrics.victims_rescued = len([v for v in self.victims if v.rescued])

        if parsed_action.type in {ActionType.SCAN_LIDAR, ActionType.SCAN_THERMAL, ActionType.SCAN_GAS, ActionType.LISTEN}:
            if self.metrics.victims_detected > prev_detected:
                self._useful_insights += self.metrics.victims_detected - prev_detected
            else:
                self._redundant_scans += 1

        if parsed_action.type == ActionType.RESCUE_VICTIM:
            if result.rescued_victim_id is not None:
                self._successful_rescues += 1
                rescued = next((v for v in self.victims if v.victim_id == result.rescued_victim_id), None)
                if rescued is not None and rescued.health == HealthStatus.CRITICAL:
                    self._correct_priority_rescues += 1
                elif had_critical_before:
                    self._incorrect_rescue_order_events += 1
            else:
                self._false_positive_detections += 1

        if self.hazard_manager.in_critical_gas_zone(self.robot.position.x, self.robot.position.y, self.hazards):
            self._critical_gas_zone_entries += 1

        done, truncated, reason = self._termination()
        if done and not self.robot.is_operational:
            self._preventable_destruction_events = 1

        current_stats = self._build_episode_stats()
        observation = self._build_observation()

        dense_reward = self._dense_step_reward(
            action=parsed_action,
            observation=observation,
            prev_detected=prev_detected,
            prev_rescued=prev_rescued,
            prev_visited=prev_visited,
            prev_nearest_victim_distance=prev_nearest_victim_distance,
            had_collision=result.collision,
            rescued_victim_id=result.rescued_victim_id,
            rejection_reason=rejection_reason,
        )

        shaped_reward = max(0.0, calculate_step_reward(previous_stats, current_stats, terminal=False))
        reward = max(0.0, min(1.0, dense_reward + 0.35 * shaped_reward))

        difficulty_factor = {
            "easy": 1.0,
            "medium": 0.92,
            "hard": 0.82,
        }.get(self.config.difficulty, 1.0)
        reward *= difficulty_factor

        progress_happened = (
            self.metrics.victims_detected > prev_detected
            or self.metrics.victims_rescued > prev_rescued
            or len(self._visited_cells) > prev_visited
            or dense_reward > 0.03
        )
        if progress_happened:
            self._no_progress_steps = 0
        else:
            self._no_progress_steps += 1

        score_breakdown = calculate_final_reward(current_stats)
        if done or truncated:
            reward = max(0.0, min(1.0, reward + 0.15 * score_breakdown.final))

        self._last_episode_stats = current_stats

        info = {
            "episode_id": self._episode_id,
            "difficulty": self.config.difficulty,
            "metrics": self.metrics.model_dump(),
            "episode_stats": current_stats.to_dict(),
            "score_breakdown": score_breakdown.to_dict(),
            "penalties": score_breakdown.penalty_events,
            "reason": reason,
            "success": done and self.metrics.victims_rescued == len(self.victims),
            "last_action_result": self._last_action_result,
            "last_action_rejected": self._last_action_rejected,
            "hint": self._last_hint,
            "no_progress_steps": self._no_progress_steps,
            "dense_reward": round(dense_reward, 4),
        }
        if done or truncated:
            info["task_grade"] = grade_episode_by_difficulty(self.config.difficulty, current_stats, score_breakdown).to_dict()
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
        explored_zones = self._sample_explored_zones(limit=12)
        unexplored_zones = self._sample_unexplored_zones(limit=12)
        hint = self._derive_hint(nearby_victims, nearby_hazards, unexplored_zones)
        self._last_hint = hint
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
            explored_zones=explored_zones,
            unexplored_zones=unexplored_zones,
            nearby_heat_signatures=thermal_signatures,
            nearby_sounds=acoustic_events,
            battery_remaining=self.robot.battery_level,
            last_action_result=self._last_action_result,
            last_action_rejected=self._last_action_rejected,
            hint=hint,
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
                self._detection_confidence_samples.append(confidence)
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

    def _build_episode_stats(self) -> EpisodeStats:
        assert self.robot is not None
        assert self.map_data is not None

        total_victims = len(self.victims)
        total_critical_victims = len([v for v in self.victims if v.health == HealthStatus.CRITICAL])
        detected_victims = len([v for v in self.victims if v.detected])
        rescued_victims = len([v for v in self.victims if v.rescued])
        critical_rescued = len([v for v in self.victims if v.rescued and v.health == HealthStatus.CRITICAL])

        total_cells = max(1, self.map_data.width * self.map_data.height)
        map_coverage = min(1.0, len(self._visited_cells) / total_cells)
        energy_used = max(0.0, 100.0 - self.robot.battery_level)
        work_accomplished = (
            2.0 * rescued_victims
            + 1.0 * self._hazards_flagged
            + 0.5 * (map_coverage * 100.0)
            - 0.3 * self._idle_steps
        )
        steps = max(1, self.metrics.steps_taken)

        detection_confidence = (
            sum(self._detection_confidence_samples) / len(self._detection_confidence_samples)
            if self._detection_confidence_samples
            else 0.0
        )

        localization_errors = []
        if detected_victims > 0:
            localization_errors = [0.4 for _ in range(min(detected_victims, 25))]

        return EpisodeStats(
            difficulty=self.config.difficulty,
            total_steps=self.metrics.steps_taken,
            collisions=self.metrics.collisions,
            collisions_near_survivor=self._collisions_near_survivor,
            joint_damage_events=0,
            tip_over_with_recovery=0,
            instability_events=self.robot.instability_events,
            successful_recoveries=self.robot.successful_recoveries,
            total_victims=total_victims,
            total_critical_victims=total_critical_victims,
            true_positives=detected_victims,
            false_positives=self._false_positive_detections,
            detection_confidence=detection_confidence,
            detected_victims=detected_victims,
            accurately_located=detected_victims,
            localization_errors_m=localization_errors,
            rescued_victims=rescued_victims,
            critical_victims_rescued=critical_rescued,
            rescue_attempts=self._rescue_attempts,
            successful_rescues=self._successful_rescues,
            improper_handling_events=0,
            dropped_victims=0,
            total_rescues=rescued_victims,
            correct_priority_rescues=self._correct_priority_rescues,
            incorrect_rescue_order_events=self._incorrect_rescue_order_events,
            correct_priority_assignments=detected_victims,
            misclassified_critical_victims=0,
            useful_insights=self._useful_insights,
            total_scans=self._scan_actions,
            coverage=map_coverage,
            missed_major_hazards=0,
            redundant_scans=self._redundant_scans,
            remaining_battery=self.robot.battery_level,
            work_accomplished=max(0.0, work_accomplished),
            energy_used=max(1.0, energy_used),
            revisit_ratio=self._revisit_steps / steps,
            idle_ratio=self._idle_steps / steps,
            smoothness=max(0.0, 1.0 - (self.metrics.collisions / steps) * 0.8),
            time_elapsed_minutes=self.metrics.time_elapsed,
            time_limit_minutes=self.profile["time_limit_minutes"],
            hazards_flagged=self._hazards_flagged,
            explored_cells=len(self._visited_cells),
            effective_actions=max(0, self.metrics.steps_taken - self._idle_steps),
            crush_injury_without_flag=0,
            critical_gas_zone_entry=self._critical_gas_zone_entries,
            false_explosion_trigger=0,
            preventable_destruction=self._preventable_destruction_events,
            map_coverage=map_coverage,
            mission_completion=(rescued_victims / max(1, total_victims)),
        )

    def _normalize_action(self, action: Action | dict[str, Any]) -> Action:
        if hasattr(action, "type"):
            return action  # type: ignore[return-value]

        if not isinstance(action, dict):
            return IdleAction()

        action_name = str(action.get("action_type") or action.get("type") or "idle")
        params = action.get("parameters", {}) if isinstance(action.get("parameters", {}), dict) else {}

        try:
            action_type = ActionType(action_name)
        except ValueError:
            action_type = ActionType.IDLE

        if action_type == ActionType.MOVE:
            target = params.get("target_position", [self.robot.position.x, self.robot.position.y] if self.robot else [0.0, 0.0])
            speed = float(params.get("speed", 0.8))
            if isinstance(target, (list, tuple)) and len(target) >= 2:
                return MoveAction(type=ActionType.MOVE, target_position=(float(target[0]), float(target[1])), speed=speed)
            return MoveAction(type=ActionType.MOVE, target_position=(0.0, 0.0), speed=speed)

        if action_type in {ActionType.SCAN_LIDAR, ActionType.SCAN_THERMAL, ActionType.SCAN_GAS, ActionType.LISTEN}:
            direction = params.get("direction")
            duration = float(params.get("duration", 1.0))
            return ScanAction(type=action_type, direction=direction, duration=duration)

        if action_type == ActionType.RESCUE_VICTIM:
            victim_id = str(params.get("victim_id", "unknown"))
            handling_method = str(params.get("handling_method", "gentle"))
            return RescueAction(type=ActionType.RESCUE_VICTIM, victim_id=victim_id, handling_method=handling_method)

        if action_type == ActionType.FLAG_HAZARD:
            hazard_type = str(params.get("hazard_type", "structural"))
            location = params.get("location", [self.robot.position.x, self.robot.position.y] if self.robot else [0.0, 0.0])
            if isinstance(location, (list, tuple)) and len(location) >= 2:
                return FlagHazardAction(
                    type=ActionType.FLAG_HAZARD,
                    hazard_type=hazard_type,
                    location=(float(location[0]), float(location[1])),
                )
            return FlagHazardAction(type=ActionType.FLAG_HAZARD, hazard_type=hazard_type, location=(0.0, 0.0))

        return IdleAction()

    def _mark_cell_visit(self, position: Position) -> None:
        cell = (int(round(position.x)), int(round(position.y)))
        if cell in self._visited_cells:
            self._revisit_steps += 1
        self._visited_cells.add(cell)

    def _apply_action_guardrails(self, action: Action) -> tuple[Action, str | None, str]:
        assert self.robot is not None

        scan_actions = {ActionType.SCAN_LIDAR, ActionType.SCAN_THERMAL, ActionType.SCAN_GAS, ActionType.LISTEN}
        if action.type in scan_actions and self.robot.battery_level <= 10.0:
            return IdleAction(duration=1.0), "low_battery_scan_blocked", "scan_blocked_low_battery"

        if action.type == ActionType.RESCUE_VICTIM and isinstance(action, RescueAction):
            if not self._is_victim_rescue_reachable(action.victim_id, max_distance=3.0):
                nearest = self._nearest_unrescued_victim()
                if nearest is not None:
                    return (
                        MoveAction(
                            type=ActionType.MOVE,
                            target_position=(nearest.position.x, nearest.position.y),
                            speed=1.1,
                        ),
                        "rescue_without_nearby_victim",
                        "rescue_blocked_move_to_nearest_victim",
                    )
                return ScanAction(type=ActionType.SCAN_THERMAL, direction=None, duration=1.0), "rescue_without_nearby_victim", "rescue_blocked_scan_thermal"

        if action.type == ActionType.MOVE and isinstance(action, MoveAction):
            target = (float(action.target_position[0]), float(action.target_position[1]))
            if self._last_move_target is not None and self._distance(self._last_move_target, target) <= 0.6:
                self._same_move_target_streak += 1
            else:
                self._same_move_target_streak = 1
            self._last_move_target = target

            if self._same_move_target_streak >= 3:
                self._same_move_target_streak = 0
                explore_target = self._next_exploration_target()
                if explore_target is not None:
                    return (
                        MoveAction(type=ActionType.MOVE, target_position=explore_target, speed=1.2),
                        "repeated_move_target",
                        "repeated_move_blocked_forced_exploration",
                    )
                return IdleAction(duration=1.0), "repeated_move_target", "repeated_move_blocked"
        else:
            self._same_move_target_streak = 0
            self._last_move_target = None

        if self._no_progress_steps >= 4 and action.type in (scan_actions | {ActionType.IDLE}):
            explore_target = self._next_exploration_target()
            if explore_target is not None:
                return (
                    MoveAction(type=ActionType.MOVE, target_position=explore_target, speed=1.2),
                    "no_progress_forced_exploration",
                    "no_progress_forced_exploration",
                )

        return action, None, "action_applied"

    def _dense_step_reward(
        self,
        action: Action,
        observation: Observation,
        prev_detected: int,
        prev_rescued: int,
        prev_visited: int,
        prev_nearest_victim_distance: float | None,
        had_collision: bool,
        rescued_victim_id: str | None,
        rejection_reason: str | None,
    ) -> float:
        dense = 0.0

        if action.type == ActionType.MOVE:
            if len(self._visited_cells) > prev_visited:
                dense += 0.05

            nearest_after = self._nearest_unrescued_victim_distance()
            if (
                prev_nearest_victim_distance is not None
                and nearest_after is not None
                and nearest_after < (prev_nearest_victim_distance - 0.25)
            ):
                dense += 0.03

            if len(self._visited_cells) == prev_visited:
                dense -= 0.02

        if action.type == ActionType.IDLE:
            dense -= 0.02

        if action.type == ActionType.SCAN_LIDAR:
            dense += 0.08 if len(self._visited_cells) > prev_visited else 0.0

        if action.type == ActionType.SCAN_THERMAL:
            new_detections = max(0, self.metrics.victims_detected - prev_detected)
            dense += 0.10 if new_detections > 0 else 0.02

        if action.type == ActionType.SCAN_GAS:
            gas = observation.sensors.gas_levels
            hazardous = gas.co > 30.0 or gas.ch4 > 8.0 or gas.h2s > 5.0
            dense += 0.15 if hazardous else 0.02

        if action.type == ActionType.LISTEN:
            dense += 0.12 if observation.nearby_sounds else 0.02

        if action.type == ActionType.RESCUE_VICTIM:
            rescued_delta = max(0, self.metrics.victims_rescued - prev_rescued)
            if rescued_delta > 0:
                dense += 0.20
                if rescued_victim_id is not None:
                    rescued = next((v for v in self.victims if v.victim_id == rescued_victim_id), None)
                    if rescued is not None and rescued.health == HealthStatus.CRITICAL:
                        dense += 0.10
            else:
                dense -= 0.02

        if action.type == ActionType.FLAG_HAZARD:
            dense += 0.15 if observation.nearby_hazards else 0.01

        if had_collision:
            dense -= 0.05

        if rejection_reason is not None:
            dense -= 0.05

        return max(0.0, min(1.0, dense))

    def _sample_explored_zones(self, limit: int = 12) -> list[tuple[int, int]]:
        assert self.robot is not None
        if not self._visited_cells:
            return []

        rx = self.robot.position.x
        ry = self.robot.position.y
        ranked = sorted(
            self._visited_cells,
            key=lambda cell: self._distance((rx, ry), (float(cell[0]), float(cell[1]))),
        )
        return [(int(c[0]), int(c[1])) for c in ranked[:limit]]

    def _sample_unexplored_zones(self, limit: int = 12) -> list[tuple[int, int]]:
        assert self.map_data is not None
        assert self.robot is not None

        candidates: list[tuple[float, tuple[int, int]]] = []
        rx = self.robot.position.x
        ry = self.robot.position.y

        for y in range(self.map_data.height):
            for x in range(self.map_data.width):
                if self.map_data.occupancy[y][x] == 1:
                    continue
                cell = (x, y)
                if cell in self._visited_cells:
                    continue
                distance = self._distance((rx, ry), (float(x), float(y)))
                candidates.append((distance, cell))

        candidates.sort(key=lambda item: item[0])
        return [(int(c[1][0]), int(c[1][1])) for c in candidates[:limit]]

    def _next_exploration_target(self) -> tuple[float, float] | None:
        assert self.robot is not None
        unexplored = self._sample_unexplored_zones(limit=20)
        if not unexplored:
            return None

        rx = self.robot.position.x
        ry = self.robot.position.y
        for cell in unexplored:
            if self._distance((rx, ry), (float(cell[0]), float(cell[1]))) > 1.0:
                return float(cell[0]), float(cell[1])
        first = unexplored[0]
        return float(first[0]), float(first[1])

    def _derive_hint(
        self,
        nearby_victims: list[VictimObservation],
        nearby_hazards: list[HazardObservation],
        unexplored_zones: list[tuple[int, int]],
    ) -> str:
        if self._last_action_rejected is not None:
            return f"Action was rejected ({self._last_action_rejected}); explore a different frontier cell."

        critical = [v for v in nearby_victims if v.health == HealthStatus.CRITICAL]
        if critical:
            target = critical[0]
            return f"Critical victim detected near ({target.position.x:.1f}, {target.position.y:.1f}); prioritize rescue."

        if nearby_victims:
            target = nearby_victims[0]
            return f"Victim signal nearby ({target.position.x:.1f}, {target.position.y:.1f}); move closer then rescue."

        if nearby_hazards:
            hazard = max(nearby_hazards, key=lambda h: h.severity)
            return f"Hazard detected ({hazard.hazard_type}); consider flagging before advancing."

        if unexplored_zones:
            zone = unexplored_zones[0]
            return f"No nearby victims; explore unexplored zone ({zone[0]}, {zone[1]})."

        if self._no_progress_steps >= 3:
            return "No progress for multiple steps; switch to thermal scan then move to a new zone."

        return "Continue mapping and use thermal/acoustic scans to discover survivors."

    def _nearest_unrescued_victim(self) -> Victim | None:
        assert self.robot is not None
        candidates = [v for v in self.victims if not v.rescued]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda v: self._distance((self.robot.position.x, self.robot.position.y), (v.position.x, v.position.y)),
        )

    def _nearest_unrescued_victim_distance(self) -> float | None:
        nearest = self._nearest_unrescued_victim()
        if nearest is None or self.robot is None:
            return None
        return self._distance((self.robot.position.x, self.robot.position.y), (nearest.position.x, nearest.position.y))

    def _is_victim_rescue_reachable(self, victim_id: str, max_distance: float) -> bool:
        assert self.robot is not None
        victim = next((v for v in self.victims if v.victim_id == victim_id and not v.rescued), None)
        if victim is None:
            return False
        dist = self._distance((self.robot.position.x, self.robot.position.y), (victim.position.x, victim.position.y))
        return dist <= max_distance

    @staticmethod
    def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _is_near_any_unrescued_victim(self, distance_threshold: float) -> bool:
        assert self.robot is not None
        for victim in self.victims:
            if victim.rescued:
                continue
            dist = ((victim.position.x - self.robot.position.x) ** 2 + (victim.position.y - self.robot.position.y) ** 2) ** 0.5
            if dist <= distance_threshold:
                return True
        return False

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

from __future__ import annotations

import json
import math
import os
import re
import sys
import textwrap
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from openai import OpenAI

from rescue_env.core.environment import RescueEnvironment

TASKS: list[tuple[str, str]] = [
    ("sweep_and_map", "easy"),
    ("strategic_triage", "medium"),
    ("extreme_rescue", "hard"),
]

DEFAULT_BENCHMARK = "rescue_robot_earthquake"
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-72B-Instruct"
DEFAULT_MAX_STEPS = 25
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_TOKENS = 180
DEFAULT_SUCCESS_THRESHOLD = 0.55
DEFAULT_BASE_SEED = 42

LOW_BATTERY_THRESHOLD = 10.0
NO_PROGRESS_FALLBACK_TRIGGER = 3
NO_PROGRESS_EXPLORE_TRIGGER = 4
FAILED_RESCUE_RETRY_LIMIT = 2
POSITION_CHANGE_EPS = 1.25
SCAN_ACTIONS = {"scan_lidar", "scan_thermal", "scan_gas", "listen"}

EXPLORATION_OFFSETS: list[tuple[float, float]] = [
    (4.0, 0.0),
    (0.0, 4.0),
    (-4.0, 0.0),
    (0.0, -4.0),
    (3.5, 3.5),
    (-3.5, 3.5),
    (-3.5, -3.5),
    (3.5, -3.5),
]


@dataclass
class VictimMemory:
    victim_id: str
    last_seen_position: tuple[float, float] | None = None
    last_seen_step: int = 0
    successful_rescue: bool = False
    failed_rescue_attempts: int = 0
    rescue_cooldown_until_step: int = 0


@dataclass
class EpisodeMemory:
    no_progress_steps: int = 0
    fallback_stage: int = 0
    victims_detected_prev: int = 0
    victims_rescued_prev: int = 0
    mission_progress_prev: float = 0.0
    last_move_target: tuple[float, float] | None = None
    same_move_target_streak: int = 0
    action_history: deque[str] = field(default_factory=lambda: deque(maxlen=20))
    frontier_cache: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=40))
    victim_table: dict[str, VictimMemory] = field(default_factory=dict)


def _get_or_create_victim_memory(memory: EpisodeMemory, victim_id: str) -> VictimMemory:
    if victim_id not in memory.victim_table:
        memory.victim_table[victim_id] = VictimMemory(victim_id=victim_id)
    return memory.victim_table[victim_id]


def _record_victim_sighting(memory: EpisodeMemory, victim_id: str, position: tuple[float, float] | None, step_index: int) -> None:
    victim = _get_or_create_victim_memory(memory, victim_id)
    victim.last_seen_step = step_index
    if position is not None:
        victim.last_seen_position = position


def _record_rescue_outcome(memory: EpisodeMemory, victim_id: str, rescue_succeeded: bool, step_index: int) -> None:
    victim = _get_or_create_victim_memory(memory, victim_id)
    if rescue_succeeded:
        victim.successful_rescue = True
        victim.failed_rescue_attempts = 0
        victim.rescue_cooldown_until_step = 0
        return

    victim.failed_rescue_attempts += 1
    victim.rescue_cooldown_until_step = step_index + min(8, 2 + victim.failed_rescue_attempts)


def _victim_on_cooldown(memory: EpisodeMemory, victim_id: str, step_index: int) -> bool:
    victim = memory.victim_table.get(victim_id)
    if victim is None:
        return False
    return step_index < victim.rescue_cooldown_until_step


def _recent_failed_victim_ids_from_memory(memory: EpisodeMemory, limit: int = 5) -> list[str]:
    ranked = sorted(
        (
            (victim_id, vm.failed_rescue_attempts)
            for victim_id, vm in memory.victim_table.items()
            if vm.failed_rescue_attempts > 0 and not vm.successful_rescue
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return [victim_id for victim_id, _ in ranked[:limit]]


def _update_frontier_cache(memory: EpisodeMemory, observation: dict[str, Any]) -> None:
    zones = observation.get("unexplored_zones", []) if isinstance(observation, dict) else []
    if not isinstance(zones, list):
        return

    for zone in zones[:12]:
        point = _extract_xy(zone)
        if point is None:
            continue
        if point in memory.frontier_cache:
            continue
        memory.frontier_cache.append(point)


def _memory_debug_snapshot(memory: EpisodeMemory) -> dict[str, Any]:
    return {
        "no_progress_steps": memory.no_progress_steps,
        "fallback_stage": memory.fallback_stage,
        "same_move_target_streak": memory.same_move_target_streak,
        "known_victims": len(memory.victim_table),
        "recent_failed_victims": _recent_failed_victim_ids_from_memory(memory, limit=4),
        "frontier_cache_size": len(memory.frontier_cache),
    }


def _load_dotenv_file(path: str) -> None:
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _append_error(existing: Optional[str], new_error: str) -> str:
    if not existing or existing == "null":
        return new_error
    if new_error in existing:
        return existing
    return f"{existing}|{new_error}"


def _sanitize_error(text: str) -> str:
    cleaned = re.sub(r"\s+", "_", text.strip())
    return cleaned[:120] if cleaned else "null"


def _action_repr(action: dict[str, Any]) -> str:
    action_type = str(action.get("action_type", "idle"))
    params = action.get("parameters", {})
    if not isinstance(params, dict):
        params = {}
    params_str = json.dumps(params, separators=(",", ":"), sort_keys=True)
    return f"{action_type}({params_str})"


def _extract_json_object(text: str) -> dict[str, Any] | None:
    payload = text.strip()
    try:
        parsed = json.loads(payload)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{[\s\S]*\}", payload)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    return parsed if isinstance(parsed, dict) else None


def _extract_xy(position: Any) -> tuple[float, float] | None:
    if isinstance(position, dict):
        try:
            return float(position.get("x", 0.0)), float(position.get("y", 0.0))
        except (TypeError, ValueError):
            return None

    if isinstance(position, (list, tuple)) and len(position) >= 2:
        try:
            return float(position[0]), float(position[1])
        except (TypeError, ValueError):
            return None

    return None


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def _robot_xy(observation: dict[str, Any]) -> tuple[float, float]:
    robot = observation.get("robot_status", {}) if isinstance(observation, dict) else {}
    pos = _extract_xy(robot.get("position") if isinstance(robot, dict) else None)
    if pos is None:
        return 0.0, 0.0
    return pos


def _battery_level(observation: dict[str, Any]) -> float:
    robot = observation.get("robot_status", {}) if isinstance(observation, dict) else {}
    if not isinstance(robot, dict):
        return 100.0
    try:
        return float(robot.get("battery_level", 100.0))
    except (TypeError, ValueError):
        return 100.0


def _nearby_victim_entries(observation: dict[str, Any]) -> list[dict[str, Any]]:
    robot_pos = _robot_xy(observation)
    victims_raw = observation.get("nearby_victims", []) if isinstance(observation, dict) else []
    entries: list[dict[str, Any]] = []

    for victim in victims_raw:
        if not isinstance(victim, dict):
            continue
        victim_id = str(victim.get("victim_id", "unknown"))
        victim_pos = _extract_xy(victim.get("position"))
        if victim_pos is None:
            continue
        entries.append(
            {
                "victim_id": victim_id,
                "position": victim_pos,
                "distance": _distance(robot_pos, victim_pos),
            }
        )

    entries.sort(key=lambda item: float(item.get("distance", 9999.0)))
    return entries


def _nearby_hazard_entries(observation: dict[str, Any]) -> list[dict[str, Any]]:
    hazards_raw = observation.get("nearby_hazards", []) if isinstance(observation, dict) else []
    entries: list[dict[str, Any]] = []
    for hazard in hazards_raw:
        if not isinstance(hazard, dict):
            continue
        pos = _extract_xy(hazard.get("position"))
        if pos is None:
            continue
        entries.append(
            {
                "hazard_type": str(hazard.get("hazard_type", "structural")),
                "severity": float(hazard.get("severity", 0.0) or 0.0),
                "position": pos,
            }
        )
    entries.sort(key=lambda item: float(item.get("severity", 0.0)), reverse=True)
    return entries


def _nearest_victim_distance(observation: dict[str, Any]) -> float | None:
    victims = _nearby_victim_entries(observation)
    if not victims:
        return None
    return float(victims[0]["distance"])


def _move_to_position_action(position: tuple[float, float], speed: float = 1.1) -> dict[str, Any]:
    return {
        "action_type": "move",
        "parameters": {
            "target_position": [float(position[0]), float(position[1])],
            "speed": max(0.2, min(1.6, float(speed))),
        },
    }


def _exploration_move_action(observation: dict[str, Any], step_index: int, no_progress_counter: int) -> dict[str, Any]:
    robot_x, robot_y = _robot_xy(observation)
    offset = EXPLORATION_OFFSETS[(step_index + no_progress_counter) % len(EXPLORATION_OFFSETS)]
    radius_scale = 1.0 + min(0.8, no_progress_counter * 0.1)
    target = (robot_x + offset[0] * radius_scale, robot_y + offset[1] * radius_scale)
    return _move_to_position_action(target, speed=1.2)


def _fallback_policy_action(
    observation: dict[str, Any],
    fallback_stage: int,
    step_index: int,
    no_progress_counter: int,
) -> dict[str, Any]:
    victims = _nearby_victim_entries(observation)
    nearest_victim = victims[0] if victims else None
    hazards = _nearby_hazard_entries(observation)
    stage = fallback_stage % 4

    if stage == 0:
        return {"action_type": "scan_thermal", "parameters": {}}

    if stage == 1:
        if nearest_victim is not None:
            return _move_to_position_action(nearest_victim["position"], speed=1.1)
        return _exploration_move_action(observation, step_index=step_index, no_progress_counter=no_progress_counter)

    if stage == 2:
        if nearest_victim is not None:
            return {
                "action_type": "rescue_victim",
                "parameters": {
                    "victim_id": nearest_victim["victim_id"],
                    "handling_method": "gentle",
                },
            }
        if hazards:
            h0 = hazards[0]
            return {
                "action_type": "flag_hazard",
                "parameters": {
                    "hazard_type": h0["hazard_type"],
                    "location": [h0["position"][0], h0["position"][1]],
                },
            }
        return _exploration_move_action(observation, step_index=step_index, no_progress_counter=no_progress_counter)

    if hazards:
        h0 = hazards[0]
        return {
            "action_type": "flag_hazard",
            "parameters": {
                "hazard_type": h0["hazard_type"],
                "location": [h0["position"][0], h0["position"][1]],
            },
        }

    if nearest_victim is not None:
        return _move_to_position_action(nearest_victim["position"], speed=1.0)

    return {"action_type": "scan_lidar", "parameters": {}}


def _retry_blocked(
    memory: EpisodeMemory,
    victim_id: str,
    current_position: tuple[float, float] | None,
    step_index: int,
) -> bool:
    victim = memory.victim_table.get(victim_id)
    if victim is None:
        return False

    if _victim_on_cooldown(memory, victim_id=victim_id, step_index=step_index):
        return True

    if victim.failed_rescue_attempts < FAILED_RESCUE_RETRY_LIMIT:
        return False

    if current_position is not None and victim.last_seen_position is not None:
        if _distance(current_position, victim.last_seen_position) > POSITION_CHANGE_EPS:
            return False

    return True


def _apply_guardrails(
    action: dict[str, Any],
    observation: dict[str, Any],
    no_progress_counter: int,
    memory: EpisodeMemory,
    step_index: int,
    history: list[str],
) -> tuple[dict[str, Any], str | None]:
    action_type = str(action.get("action_type", "idle"))
    victims = _nearby_victim_entries(observation)
    nearest_victim = victims[0] if victims else None
    battery = _battery_level(observation)

    proposed_repr = _action_repr(action)
    if len(history) >= 3 and all(item == history[-1] for item in history[-3:]) and proposed_repr == history[-1]:
        return _exploration_move_action(observation, step_index=step_index, no_progress_counter=max(3, no_progress_counter)), "guardrail_repeated_action_loop"

    scan_like_actions = {"scan_lidar", "scan_thermal", "scan_gas", "listen"}
    if len(history) >= 2 and history[-1] == history[-2] == proposed_repr and proposed_repr in scan_like_actions and no_progress_counter >= 2:
        return _exploration_move_action(observation, step_index=step_index, no_progress_counter=max(3, no_progress_counter)), "guardrail_repeated_scan_loop"

    if no_progress_counter >= NO_PROGRESS_EXPLORE_TRIGGER:
        return _exploration_move_action(observation, step_index=step_index, no_progress_counter=no_progress_counter), "guardrail_forced_explore_no_progress"

    if action_type in SCAN_ACTIONS and battery <= LOW_BATTERY_THRESHOLD:
        if nearest_victim is not None:
            return _move_to_position_action(nearest_victim["position"], speed=1.0), "guardrail_low_battery_scan_blocked"
        return {"action_type": "idle", "parameters": {}}, "guardrail_low_battery_idle"

    if action_type == "rescue_victim":
        params = action.get("parameters", {}) if isinstance(action.get("parameters"), dict) else {}
        target_victim_id = str(params.get("victim_id", "unknown"))
        target_entry = next((item for item in victims if item["victim_id"] == target_victim_id), None)

        if not victims:
            if battery <= LOW_BATTERY_THRESHOLD:
                return {"action_type": "idle", "parameters": {}}, "guardrail_rescue_without_nearby_victim"
            return {"action_type": "scan_thermal", "parameters": {}}, "guardrail_rescue_without_nearby_victim"

        if target_entry is None and nearest_victim is not None:
            return _move_to_position_action(nearest_victim["position"], speed=1.1), "guardrail_rescue_target_not_nearby"

        if target_entry is not None and _retry_blocked(
            memory=memory,
            victim_id=target_victim_id,
            current_position=target_entry["position"],
            step_index=step_index,
        ):
            return _exploration_move_action(observation, step_index=step_index, no_progress_counter=no_progress_counter), "guardrail_rescue_retry_blocked"

    if action_type == "move":
        params = action.get("parameters", {}) if isinstance(action.get("parameters"), dict) else {}
        target_position = _extract_xy(params.get("target_position"))
        if target_position is not None:
            robot_position = _robot_xy(observation)
            if _distance(target_position, robot_position) < 0.6 and no_progress_counter >= 2:
                return _exploration_move_action(observation, step_index=step_index, no_progress_counter=max(3, no_progress_counter)), "guardrail_zero_delta_move"

    if no_progress_counter >= NO_PROGRESS_FALLBACK_TRIGGER and action_type == "idle":
        if nearest_victim is not None:
            return _move_to_position_action(nearest_victim["position"], speed=1.0), "guardrail_idle_no_progress"
        return {"action_type": "scan_thermal", "parameters": {}}, "guardrail_scan_no_progress"

    return action, None


def _normalize_action(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"action_type": "idle", "parameters": {}}

    action_type = str(raw.get("action_type") or raw.get("type") or "idle")
    parameters = raw.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    if action_type == "move":
        target = parameters.get("target_position", [0.0, 0.0])
        if not isinstance(target, (list, tuple)) or len(target) < 2:
            target = [0.0, 0.0]
        speed = float(parameters.get("speed", 0.8))
        return {
            "action_type": "move",
            "parameters": {
                "target_position": [float(target[0]), float(target[1])],
                "speed": max(0.0, min(1.6, speed)),
            },
        }

    if action_type == "rescue_victim":
        return {
            "action_type": "rescue_victim",
            "parameters": {
                "victim_id": str(parameters.get("victim_id", "unknown")),
                "handling_method": str(parameters.get("handling_method", "gentle")),
            },
        }

    if action_type == "flag_hazard":
        location = parameters.get("location", [0.0, 0.0])
        if not isinstance(location, (list, tuple)) or len(location) < 2:
            location = [0.0, 0.0]
        return {
            "action_type": "flag_hazard",
            "parameters": {
                "hazard_type": str(parameters.get("hazard_type", "structural")),
                "location": [float(location[0]), float(location[1])],
            },
        }

    allowed = {"scan_lidar", "scan_thermal", "scan_gas", "listen", "idle"}
    if action_type in allowed:
        return {"action_type": action_type, "parameters": {}}

    return {"action_type": "idle", "parameters": {}}


def _build_user_prompt(
    step_index: int,
    difficulty: str,
    task_name: str,
    observation: dict[str, Any],
    last_reward: float,
    history: list[str],
    failed_victim_ids: list[str],
    no_progress_counter: int,
    nearest_victim_distance: float | None,
    memory: EpisodeMemory,
) -> str:
    robot = observation.get("robot_status", {}) if isinstance(observation, dict) else {}
    nearby_victims = observation.get("nearby_victims", []) if isinstance(observation, dict) else []
    nearby_hazards = observation.get("nearby_hazards", []) if isinstance(observation, dict) else []

    summary = {
        "step": step_index,
        "difficulty": difficulty,
        "task_name": task_name,
        "robot": {
            "position": robot.get("position"),
            "battery_level": robot.get("battery_level"),
            "is_stable": robot.get("is_stable"),
        },
        "nearby_counts": {
            "victims": len(nearby_victims),
            "hazards": len(nearby_hazards),
        },
        "mission_progress": observation.get("mission_progress"),
        "time_remaining": observation.get("time_remaining"),
        "last_reward": round(last_reward, 4),
        "last_5_actions": history[-5:],
        "recently_failed_victim_ids": failed_victim_ids,
        "no_progress_counter": no_progress_counter,
        "nearest_victim_distance_m": None if nearest_victim_distance is None else round(nearest_victim_distance, 3),
        "nearby_victim_ids": [
            str(v.get("victim_id", "unknown"))
            for v in nearby_victims
            if isinstance(v, dict)
        ],
        "frontier_cache": [[round(x, 2), round(y, 2)] for x, y in list(memory.frontier_cache)[-6:]],
        "memory_snapshot": _memory_debug_snapshot(memory),
    }

    return textwrap.dedent(
        f"""
        You control a rescue robot in an earthquake site.
        Pick exactly one next action.

        Return strict JSON object only:
        {{"action_type":"<one of move|scan_lidar|scan_thermal|scan_gas|listen|rescue_victim|flag_hazard|idle>","parameters":{{...}}}}

        Parameter rules:
        - move: parameters.target_position=[x,y], parameters.speed in [0.0,1.6]
        - rescue_victim: parameters.victim_id, parameters.handling_method
        - flag_hazard: parameters.hazard_type, parameters.location=[x,y]
        - scan/listen/idle: parameters={{}}

        Important policy constraints:
        - Avoid repeating rescue attempts on the same victim unless position context changes.
        - If no progress is happening, favor exploration moves or thermal scan before re-attempting rescue.
        - If battery is low, avoid unnecessary scans.

        Observation summary:
        {json.dumps(summary, separators=(",", ":"), default=str)}
        """
    ).strip()


def _call_model(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    seed: int,
    temperature: float,
    max_tokens: int,
) -> str:
    request_payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "seed": seed,
    }

    try:
        response = client.chat.completions.create(**request_payload)
    except Exception as exc:  # noqa: BLE001
        err_text = str(exc).lower()
        # Some OpenAI-compatible providers (including Gemini adapters) reject `seed`.
        if "unknown name \"seed\"" not in err_text and "invalid json payload" not in err_text:
            raise
        request_payload.pop("seed", None)
        response = client.chat.completions.create(**request_payload)

    content = response.choices[0].message.content
    return content if isinstance(content, str) else str(content)


def _run_one_task(
    client: OpenAI,
    task_name: str,
    difficulty: str,
    model_name: str,
    benchmark_name: str,
    seed: int,
    max_steps: int,
    temperature: float,
    max_tokens: int,
    success_threshold: float,
    memory_debug: bool,
) -> float:
    env = RescueEnvironment(difficulty=difficulty)
    observation = env.reset(seed=seed)

    rewards: list[float] = []
    history: list[str] = []
    info: dict[str, Any] = {}
    memory = EpisodeMemory()

    system_prompt = (
        "You are an autonomous rescue policy. "
        "Maximize rescues, prioritize critical victims, avoid hazards, and avoid wasting steps. "
        "Always return valid JSON only. "
        "Do not spam rescue on the same victim without new context."
    )

    log_start(task_name, benchmark_name, model_name)

    last_reward = 0.0
    for step_index in range(1, max_steps + 1):
        obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else dict(observation)
        _update_frontier_cache(memory, obs_dict)
        nearest_dist = _nearest_victim_distance(obs_dict)
        failed_ids = _recent_failed_victim_ids_from_memory(memory, limit=5)
        user_prompt = _build_user_prompt(
            step_index,
            difficulty,
            task_name,
            obs_dict,
            last_reward,
            history,
            failed_victim_ids=failed_ids,
            no_progress_counter=memory.no_progress_steps,
            nearest_victim_distance=nearest_dist,
            memory=memory,
        )

        model_error: Optional[str] = None
        use_fallback_policy = False
        try:
            response_text = _call_model(
                client=client,
                model=model_name,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                seed=seed + step_index,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            action = _normalize_action(_extract_json_object(response_text))
            if action["action_type"] == "idle" and "idle" not in response_text.lower():
                model_error = "invalid_json_fallback_idle"
                use_fallback_policy = True
        except Exception as exc:  # noqa: BLE001
            model_error = _sanitize_error(str(exc))
            action = {"action_type": "idle", "parameters": {}}
            use_fallback_policy = True

        if memory.no_progress_steps >= NO_PROGRESS_FALLBACK_TRIGGER:
            use_fallback_policy = True

        if use_fallback_policy:
            action = _fallback_policy_action(
                observation=obs_dict,
                fallback_stage=memory.fallback_stage,
                step_index=step_index,
                no_progress_counter=memory.no_progress_steps,
            )
            memory.fallback_stage += 1
            model_error = _append_error(model_error, "fallback_policy_applied")

        action, guardrail_reason = _apply_guardrails(
            action=action,
            observation=obs_dict,
            no_progress_counter=memory.no_progress_steps,
            memory=memory,
            step_index=step_index,
            history=history,
        )
        if guardrail_reason:
            model_error = _append_error(model_error, guardrail_reason)

        if action.get("action_type") == "move":
            params = action.get("parameters", {}) if isinstance(action.get("parameters"), dict) else {}
            move_target = _extract_xy(params.get("target_position"))
            if move_target is not None:
                if memory.last_move_target is not None and _distance(memory.last_move_target, move_target) < 0.5:
                    memory.same_move_target_streak += 1
                else:
                    memory.same_move_target_streak = 1
                memory.last_move_target = move_target
            else:
                memory.same_move_target_streak = 0
                memory.last_move_target = None

            if memory.same_move_target_streak >= 3:
                action = _exploration_move_action(observation=obs_dict, step_index=step_index, no_progress_counter=max(3, memory.no_progress_steps))
                model_error = _append_error(model_error, "guardrail_repeated_move_target")
                memory.same_move_target_streak = 0
                memory.last_move_target = None
        else:
            memory.same_move_target_streak = 0
            memory.last_move_target = None

        done = False
        truncated = False
        reward = 0.0

        rescued_before = memory.victims_rescued_prev
        detected_before = memory.victims_detected_prev
        mission_before = memory.mission_progress_prev

        target_victim_id = None
        target_victim_position = None
        if action.get("action_type") == "rescue_victim":
            params = action.get("parameters", {}) if isinstance(action.get("parameters"), dict) else {}
            target_victim_id = str(params.get("victim_id", "unknown"))
            victims_now = _nearby_victim_entries(obs_dict)
            target_entry = next((item for item in victims_now if item["victim_id"] == target_victim_id), None)
            if target_entry is not None:
                target_victim_position = target_entry["position"]
                _record_victim_sighting(memory, target_victim_id, target_victim_position, step_index)

        try:
            observation, reward, done, truncated, info = env.step(action)
        except Exception as exc:  # noqa: BLE001
            model_error = _sanitize_error(f"step_error:{exc}")
            fallback = _fallback_policy_action(
                observation=obs_dict,
                fallback_stage=memory.fallback_stage,
                step_index=step_index,
                no_progress_counter=memory.no_progress_steps,
            )
            memory.fallback_stage += 1
            observation, reward, done, truncated, info = env.step(fallback)
            action = fallback
            model_error = _append_error(model_error, "fallback_policy_after_step_error")

        reward_for_log = _clip01(reward)
        rewards.append(reward_for_log)
        last_reward = reward_for_log

        metrics_payload = info.get("metrics", {}) if isinstance(info, dict) else {}
        if isinstance(metrics_payload, dict):
            victims_detected_after = int(metrics_payload.get("victims_detected", detected_before) or detected_before)
            victims_rescued_after = int(metrics_payload.get("victims_rescued", rescued_before) or rescued_before)
        else:
            victims_detected_after = detected_before
            victims_rescued_after = rescued_before

        obs_after_dict = observation.model_dump() if hasattr(observation, "model_dump") else dict(observation)
        try:
            mission_after = float(obs_after_dict.get("mission_progress", mission_before) or mission_before)
        except (TypeError, ValueError):
            mission_after = mission_before

        progress_happened = (
            victims_rescued_after > rescued_before
            or victims_detected_after > detected_before
            or mission_after > (mission_before + 1e-6)
            or reward_for_log > 0.0
        )

        if progress_happened:
            memory.no_progress_steps = 0
        else:
            memory.no_progress_steps += 1

        memory.victims_detected_prev = victims_detected_after
        memory.victims_rescued_prev = victims_rescued_after
        memory.mission_progress_prev = mission_after

        if action.get("action_type") == "rescue_victim" and target_victim_id is not None:
            rescue_succeeded = victims_rescued_after > rescued_before
            _record_victim_sighting(memory, target_victim_id, target_victim_position, step_index)
            _record_rescue_outcome(memory, target_victim_id, rescue_succeeded, step_index)

        action_repr = _action_repr(action)
        done_flag = bool(done or truncated)
        log_step(
            step=step_index,
            action=action_repr,
            reward=reward_for_log,
            done=done_flag,
            error=model_error,
        )
        if memory_debug:
            print(f"[MEMORY] {json.dumps(_memory_debug_snapshot(memory), separators=(',', ':'))}", flush=True)

        history.append(action_repr)
        memory.action_history.append(action_repr)

        if done_flag:
            break

    task_grade = info.get("task_grade") if isinstance(info, dict) else None
    score = 0.0
    success = False

    if isinstance(task_grade, dict):
        score = float(task_grade.get("task_score", task_grade.get("reward", 0.0)))
        success = bool(task_grade.get("success", False))
    else:
        breakdown = info.get("score_breakdown") if isinstance(info, dict) else None
        if isinstance(breakdown, dict):
            score = float(breakdown.get("final", 0.0))

    score = _clip01(score)
    if not success:
        success = score >= success_threshold

    log_end(success=success, steps=len(rewards), score=score, rewards=rewards)
    return score


def main() -> int:
    project_root = os.path.dirname(os.path.abspath(__file__))
    _load_dotenv_file(os.path.join(project_root, ".env"))

    benchmark_name = os.getenv("BENCHMARK", DEFAULT_BENCHMARK)
    base_url = os.getenv("API_BASE_URL", "").strip()
    model_name = os.getenv("MODEL_NAME", "").strip()
    max_steps = int(os.getenv("MAX_STEPS", str(DEFAULT_MAX_STEPS)))
    temperature = float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    max_tokens = int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    success_threshold = float(os.getenv("SUCCESS_SCORE_THRESHOLD", str(DEFAULT_SUCCESS_THRESHOLD)))
    base_seed = int(os.getenv("BASE_SEED", str(DEFAULT_BASE_SEED)))
    memory_debug = os.getenv("MEMORY_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

    if not base_url:
        print("Missing required variable API_BASE_URL.", flush=True)
        return 1

    if not model_name:
        print("Missing required variable MODEL_NAME.", flush=True)
        return 1

    openai_api_key = os.getenv("OPENAI_API_KEY")
    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        print("Missing required variable HF_TOKEN.", flush=True)
        return 1

    api_key = openai_api_key or hf_token or os.getenv("API_KEY")
    if not api_key:
        print("Missing credentials. Set OPENAI_API_KEY or API_KEY (HF_TOKEN is also required by submission).", flush=True)
        return 1

    client = OpenAI(api_key=api_key, base_url=base_url)

    selected_task = os.getenv("TASK_NAME", "all").strip().lower()
    all_scores: list[float] = []

    for index, (task_name, difficulty) in enumerate(TASKS):
        if selected_task != "all" and selected_task not in {task_name, difficulty}:
            continue
        task_seed = base_seed + index * 1000
        score = _run_one_task(
            client=client,
            task_name=task_name,
            difficulty=difficulty,
            model_name=model_name,
            benchmark_name=benchmark_name,
            seed=task_seed,
            max_steps=max_steps,
            temperature=temperature,
            max_tokens=max_tokens,
            success_threshold=success_threshold,
            memory_debug=memory_debug,
        )
        all_scores.append(score)

    if not all_scores:
        print("No tasks selected. Set TASK_NAME to all/easy/medium/hard or a task name.", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

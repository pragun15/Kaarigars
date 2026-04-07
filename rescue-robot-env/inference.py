from __future__ import annotations

import json
import os
import re
import sys
import textwrap
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
        "recent_actions": history[-4:],
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
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
    )
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
) -> float:
    env = RescueEnvironment(difficulty=difficulty)
    observation = env.reset(seed=seed)

    rewards: list[float] = []
    history: list[str] = []
    info: dict[str, Any] = {}

    system_prompt = (
        "You are an autonomous rescue policy. "
        "Maximize rescues, prioritize critical victims, avoid hazards, and avoid wasting steps. "
        "Always return valid JSON only."
    )

    log_start(task_name, benchmark_name, model_name)

    last_reward = 0.0
    for step_index in range(1, max_steps + 1):
        obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else dict(observation)
        user_prompt = _build_user_prompt(step_index, difficulty, task_name, obs_dict, last_reward, history)

        model_error: Optional[str] = None
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
        except Exception as exc:  # noqa: BLE001
            model_error = _sanitize_error(str(exc))
            action = {"action_type": "idle", "parameters": {}}

        done = False
        truncated = False
        reward = 0.0

        try:
            observation, reward, done, truncated, info = env.step(action)
        except Exception as exc:  # noqa: BLE001
            model_error = _sanitize_error(f"step_error:{exc}")
            fallback = {"action_type": "idle", "parameters": {}}
            observation, reward, done, truncated, info = env.step(fallback)
            action = fallback

        reward_for_log = _clip01(reward)
        rewards.append(reward_for_log)
        last_reward = reward_for_log

        action_repr = _action_repr(action)
        done_flag = bool(done or truncated)
        log_step(
            step=step_index,
            action=action_repr,
            reward=reward_for_log,
            done=done_flag,
            error=model_error,
        )

        history.append(action_repr)

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
    base_url = os.getenv("API_BASE_URL", DEFAULT_API_BASE_URL)
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    max_steps = int(os.getenv("MAX_STEPS", str(DEFAULT_MAX_STEPS)))
    temperature = float(os.getenv("TEMPERATURE", str(DEFAULT_TEMPERATURE)))
    max_tokens = int(os.getenv("MAX_TOKENS", str(DEFAULT_MAX_TOKENS)))
    success_threshold = float(os.getenv("SUCCESS_SCORE_THRESHOLD", str(DEFAULT_SUCCESS_THRESHOLD)))
    base_seed = int(os.getenv("BASE_SEED", str(DEFAULT_BASE_SEED)))

    hf_token = os.getenv("HF_TOKEN")
    api_key = hf_token or os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing credentials. Set HF_TOKEN (preferred) or API_KEY/OPENAI_API_KEY.", flush=True)
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
        )
        all_scores.append(score)

    if not all_scores:
        print("No tasks selected. Set TASK_NAME to all/easy/medium/hard or a task name.", flush=True)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

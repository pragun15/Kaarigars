from __future__ import annotations

import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any

# Allow running this script via absolute path from any working directory.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from rescue_env.core.environment import RescueEnvironment


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Try to parse a JSON object from model output text."""
    text = text.strip()

    # Fast path: whole text is JSON.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Fallback: grab first object-looking block.
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, dict):
        return parsed
    return None


def _normalize_action(candidate: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(candidate, dict):
        return {"action_type": "idle", "parameters": {}}

    action_type = candidate.get("action_type") or candidate.get("type") or "idle"
    parameters = candidate.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}

    return {"action_type": str(action_type), "parameters": parameters}


def call_openai_compatible(
    api_key: str,
    base_url: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int = 60,
    max_retries: int = 2,
) -> str:
    endpoint = base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0,
    }

    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                response_body = response.read().decode("utf-8")
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            retry_after = _extract_retry_delay_seconds(body, exc.headers)
            exhausted = _quota_exhausted_message(body)

            if exhausted:
                raise RuntimeError(exhausted) from exc

            is_retryable = exc.code in {408, 409, 425, 429, 500, 502, 503, 504}
            if is_retryable and attempt < max_retries:
                print(f"API temporary error {exc.code}. Retrying in {retry_after:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_after)
                continue

            raise RuntimeError(f"API HTTP error {exc.code}: {body}") from exc
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt < max_retries:
                retry_after = 2.0 + attempt
                print(f"API connection error. Retrying in {retry_after:.1f}s (attempt {attempt + 1}/{max_retries})...")
                time.sleep(retry_after)
                continue
            raise RuntimeError(f"API connection error: {exc}") from exc
    else:
        if last_error is not None:
            raise RuntimeError(f"API connection error: {last_error}") from last_error
        raise RuntimeError("API call failed without a specific error.")

    parsed = json.loads(response_body)
    try:
        return parsed["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected API response: {parsed}") from exc


def build_user_prompt(observation: dict[str, Any]) -> str:
    robot = observation.get("robot_status", {})
    victims = observation.get("nearby_victims", [])
    hazards = observation.get("nearby_hazards", [])

    summary = {
        "robot": {
            "position": robot.get("position"),
            "battery_level": robot.get("battery_level"),
            "is_stable": robot.get("is_stable"),
        },
        "counts": {
            "nearby_victims": len(victims),
            "nearby_hazards": len(hazards),
        },
        "time_remaining": observation.get("time_remaining"),
        "mission_progress": observation.get("mission_progress"),
    }

    return (
        "Choose exactly one next action for earthquake rescue. "
        "Return only JSON with keys action_type and parameters. "
        "Allowed action_type values: move, scan_lidar, scan_thermal, scan_gas, listen, rescue_victim, flag_hazard, idle. "
        "If move, include parameters.target_position [x, y] and parameters.speed. "
        "If rescue_victim, include parameters.victim_id and parameters.handling_method. "
        "If flag_hazard, include parameters.hazard_type and parameters.location [x, y]. "
        f"Observation summary: {json.dumps(summary)}"
    )


def _load_dotenv_file(path: str) -> None:
    """Load KEY=VALUE pairs from a .env file into process env without overwriting existing values."""
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                continue

            os.environ.setdefault(key, value)


def _extract_retry_delay_seconds(body: str, headers: Any) -> float:
    """Parse retry delay from HTTP headers/body; fallback to a small default."""
    retry_after_header = None
    if headers is not None:
        try:
            retry_after_header = headers.get("Retry-After")
        except Exception:
            retry_after_header = None

    if retry_after_header:
        try:
            return max(1.0, float(retry_after_header))
        except ValueError:
            pass

    # Gemini often returns retryDelay values like "18s" in error details.
    match = re.search(r'"retryDelay"\s*:\s*"([0-9]+(?:\.[0-9]+)?)s"', body)
    if match:
        try:
            return max(1.0, float(match.group(1)))
        except ValueError:
            pass

    return 3.0


def _quota_exhausted_message(body: str) -> str | None:
    lowered = body.lower()
    if "quota exceeded" in lowered or "limit: 0" in lowered or "resource_exhausted" in lowered:
        return (
            "API key is valid, but your Gemini project quota is exhausted or set to zero. "
            "Enable billing or switch to a project/key with available quota, then run again."
        )
    return None


def main() -> int:
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    _load_dotenv_file(os.path.join(project_root, ".env"))

    api_key = os.getenv("AI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Missing API key. Set AI_API_KEY or OPENAI_API_KEY.")
        return 1

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    model = os.getenv("AI_MODEL", "gpt-4o-mini")
    difficulty = os.getenv("ENV_DIFFICULTY", "easy")
    max_steps = int(os.getenv("MAX_STEPS", "10"))
    seed = int(os.getenv("SEED", "42"))
    request_timeout = int(os.getenv("REQUEST_TIMEOUT", "60"))
    max_api_retries = int(os.getenv("MAX_API_RETRIES", "2"))

    env = RescueEnvironment(difficulty=difficulty)
    observation = env.reset(seed=seed)

    system_prompt = (
        "You are controlling a rescue robot. "
        "Prioritize critical victims, avoid hazards, and conserve battery. "
        "Return only valid JSON."
    )

    done = False
    truncated = False
    info: dict[str, Any] = {}

    print(f"Running LLM smoke test with model={model}, difficulty={difficulty}, steps={max_steps}")

    for step_index in range(1, max_steps + 1):
        obs_dict = observation.model_dump() if hasattr(observation, "model_dump") else dict(observation)
        user_prompt = build_user_prompt(obs_dict)

        try:
            response_text = call_openai_compatible(
                api_key=api_key,
                base_url=base_url,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                timeout=request_timeout,
                max_retries=max_api_retries,
            )
        except RuntimeError as exc:
            print(f"LLM request failed: {exc}")
            return 2
        action = _normalize_action(_extract_json_object(response_text))

        observation, reward, done, truncated, info = env.step(action)
        print(
            f"step={step_index:02d} action={action.get('action_type')} reward={reward:.4f} "
            f"done={done} truncated={truncated}"
        )

        if done or truncated:
            break

    reason = info.get("reason")
    score = (info.get("score_breakdown") or {}).get("final")
    print(f"Finished: reason={reason} final_score={score}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

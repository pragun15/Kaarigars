from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

REQUIRED_FILES = [
    "openenv.yaml",
    "Dockerfile",
    "inference.py",
]

REQUIRED_ENV = [
    "API_BASE_URL",
    "MODEL_NAME",
    "HF_TOKEN",
]

AUTH_ERROR_PATTERNS = [
    r"api_key_not_valid",
    r"invalid api key",
    r"unauthorized",
    r"missing required variable hf_token",
    r"missing credentials",
]


def _load_dotenv(dotenv_path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not dotenv_path.exists():
        return data

    for raw in dotenv_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            data[key] = value
    return data


def _fail(msg: str) -> int:
    print(f"[FAIL] {msg}")
    return 1


def _pass(msg: str) -> None:
    print(f"[PASS] {msg}")


def _warn(msg: str) -> None:
    print(f"[WARN] {msg}")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)

    for path in REQUIRED_FILES:
        if not Path(path).exists():
            return _fail(f"Missing required file: {path}")
    _pass("Required files exist")

    dotenv = _load_dotenv(root / ".env")
    merged = os.environ.copy()
    for key, value in dotenv.items():
        merged.setdefault(key, value)

    strict_auth = merged.get("LOCAL_CHECK_STRICT_AUTH", "0").strip().lower() in {"1", "true", "yes", "on"}

    missing = [key for key in REQUIRED_ENV if not merged.get(key)]
    if missing:
        return _fail(f"Missing required environment variables: {', '.join(missing)}")
    _pass("Required environment variables are defined")

    cmd = [sys.executable, "inference.py"]
    run_env = merged.copy()
    run_env.setdefault("TASK_NAME", "all")
    run_env.setdefault("MAX_STEPS", "3")
    run_env.setdefault("TEMPERATURE", "0.0")

    proc = subprocess.run(cmd, capture_output=True, text=True, env=run_env)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")

    if proc.returncode != 0:
        print(output)
        return _fail(f"inference.py exited with code {proc.returncode}")
    _pass("inference.py exited successfully")

    starts = re.findall(r"^\[START\]\s+task=.+\s+env=.+\s+model=.+$", output, re.MULTILINE)
    steps = re.findall(r"^\[STEP\]\s+step=\d+\s+action=.+\s+reward=-?\d+(?:\.\d+)?\s+done=(?:true|false)\s+error=.*$", output, re.MULTILINE)
    ends = re.findall(r"^\[END\]\s+success=(?:true|false)\s+steps=\d+\s+score=\d+(?:\.\d+)?\s+rewards=.*$", output, re.MULTILINE)

    if len(starts) < 3:
        print(output)
        return _fail("Expected at least 3 [START] lines (one per task)")
    if len(steps) == 0:
        print(output)
        return _fail("No [STEP] lines found")
    if len(ends) < 3:
        print(output)
        return _fail("Expected at least 3 [END] lines (one per task)")
    _pass("Structured log format detected for [START]/[STEP]/[END]")

    score_matches = re.findall(r"\[END\].*?score=(\d+(?:\.\d+)?)", output)
    if len(score_matches) < 3:
        return _fail("Could not parse all task scores from [END] lines")

    for raw in score_matches:
        score = float(raw)
        if not (0.0 <= score <= 1.0):
            return _fail(f"Score out of bounds: {score}")
    _pass("All parsed task scores are within [0.0, 1.0]")

    lowered = output.lower()
    for pattern in AUTH_ERROR_PATTERNS:
        if re.search(pattern, lowered):
            if strict_auth:
                return _fail("Inference output shows authentication/config error; fix credentials before submit")
            _warn("Inference output shows authentication/config error; treated as non-blocking in local mode (set LOCAL_CHECK_STRICT_AUTH=1 to enforce)")
            break
    else:
        _pass("No obvious authentication/config errors detected in inference output")

    print("[PASS] Local pre-submit check completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

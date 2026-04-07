from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rescue_env.core.environment import RescueEnvironment


class ResetRequest(BaseModel):
    difficulty: str = Field(default="easy", pattern="^(easy|medium|hard)$")
    seed: int | None = None


class StepRequest(BaseModel):
    action: dict[str, Any]


app = FastAPI(title="Rescue Robot OpenEnv Service", version="0.1.0")

_CURRENT_DIFFICULTY = "easy"
_ENV = RescueEnvironment(difficulty=_CURRENT_DIFFICULTY)
_OBSERVATION: Any | None = None


def _obs_to_dict(observation: Any) -> dict[str, Any]:
    if hasattr(observation, "model_dump"):
        return observation.model_dump()
    if isinstance(observation, dict):
        return observation
    return {"raw": str(observation)}


@app.on_event("startup")
def startup() -> None:
    global _OBSERVATION
    _OBSERVATION = _ENV.reset(seed=42)


@app.get("/")
def root() -> dict[str, Any]:
    return {
        "status": "ok",
        "service": "rescue-robot-openenv",
        "endpoints": ["/health", "/reset", "/step", "/state"],
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "healthy"}


@app.post("/reset")
@app.post("/openenv/reset")
def reset(request: ResetRequest) -> dict[str, Any]:
    global _ENV
    global _OBSERVATION
    global _CURRENT_DIFFICULTY

    _CURRENT_DIFFICULTY = request.difficulty
    _ENV = RescueEnvironment(difficulty=request.difficulty)
    _OBSERVATION = _ENV.reset(seed=request.seed)

    return {
        "difficulty": _CURRENT_DIFFICULTY,
        "observation": _obs_to_dict(_OBSERVATION),
    }


@app.post("/step")
@app.post("/openenv/step")
def step(request: StepRequest) -> dict[str, Any]:
    global _OBSERVATION
    if _OBSERVATION is None:
        _OBSERVATION = _ENV.reset(seed=42)

    try:
        observation, reward, done, truncated, info = _ENV.step(request.action)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    _OBSERVATION = observation
    return {
        "observation": _obs_to_dict(observation),
        "reward": float(max(0.0, min(1.0, reward))),
        "done": bool(done),
        "truncated": bool(truncated),
        "info": info,
    }


@app.get("/state")
@app.get("/openenv/state")
def state() -> dict[str, Any]:
    try:
        snapshot = _ENV.state()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return snapshot.model_dump()

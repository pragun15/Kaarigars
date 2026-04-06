"""Configuration model and helpers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from rescue_env.core.constants import DIFFICULTY_PROFILES


class EnvConfig(BaseModel):
    difficulty: str = Field(default="easy", pattern="^(easy|medium|hard)$")
    max_steps: int = Field(default=1200, ge=1)
    seed: int = 0
    render_mode: str | None = None


def get_profile(difficulty: str) -> dict[str, Any]:
    if difficulty not in DIFFICULTY_PROFILES:
        raise ValueError(f"Unsupported difficulty: {difficulty}")
    return DIFFICULTY_PROFILES[difficulty].copy()

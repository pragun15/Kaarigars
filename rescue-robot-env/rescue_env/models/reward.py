"""Reward models for typed OpenEnv-compatible outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field


class Reward(BaseModel):
    """Typed reward payload with normalized scalar and optional components."""

    value: float = Field(ge=0.0, le=1.0)
    dense_component: float = Field(default=0.0, ge=0.0, le=1.0)
    shaped_component: float = Field(default=0.0, ge=0.0, le=1.0)
    terminal_component: float = Field(default=0.0, ge=0.0, le=1.0)

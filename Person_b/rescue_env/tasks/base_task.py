from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rescue_env.scoring.types import clip01


@dataclass
class TaskSpec:
    """Task definition with deterministic success evaluation."""

    name: str
    difficulty: str
    description: str
    time_limit_minutes: float
    success_criteria: dict[str, float]
    environment_profile: dict[str, Any] = field(default_factory=dict)

    def evaluate(self, measured_metrics: dict[str, float]) -> tuple[float, bool, dict[str, dict[str, float | bool]]]:
        """Return (task_score, success, per_criterion_details)."""
        details: dict[str, dict[str, float | bool]] = {}
        ratios: list[float] = []

        for metric_name, target in self.success_criteria.items():
            value = float(measured_metrics.get(metric_name, 0.0))
            ratio = clip01(value / target) if target > 0 else 1.0
            passed = value >= target
            ratios.append(ratio)
            details[metric_name] = {
                "value": value,
                "target": target,
                "ratio": ratio,
                "passed": passed,
            }

        task_score = sum(ratios) / max(1, len(ratios))
        success = all(item["passed"] for item in details.values()) if details else False
        return clip01(task_score), success, details

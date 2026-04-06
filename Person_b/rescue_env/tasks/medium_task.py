from __future__ import annotations

from rescue_env.tasks.base_task import TaskSpec


def build_medium_task() -> TaskSpec:
    return TaskSpec(
        name="strategic_triage",
        difficulty="medium",
        description="Prioritize critical victims under impaired visibility and moderate debris.",
        time_limit_minutes=60.0,
        success_criteria={
            "critical_victims_rescued": 0.80,
            "priority_score": 0.75,
            "safety_score": 0.70,
        },
        environment_profile={
            "weather": "winter_fog",
            "magnitude_range": [5.5, 6.5],
            "entrapped_rate_range": [0.20, 0.35],
            "sensor_noise": 0.20,
        },
    )

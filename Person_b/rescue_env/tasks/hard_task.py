from __future__ import annotations

from rescue_env.tasks.base_task import TaskSpec


def build_hard_task() -> TaskSpec:
    return TaskSpec(
        name="extreme_rescue",
        difficulty="hard",
        description="Mass casualty rescue under severe hazards and sensor degradation.",
        time_limit_minutes=45.0,
        success_criteria={
            "victims_rescued": 0.60,
            "decision_score": 0.65,
            "mission_completion": 0.70,
        },
        environment_profile={
            "weather": "rain_mud",
            "magnitude_range": [6.5, 7.5],
            "entrapped_rate_range": [0.50, 0.85],
            "sensor_noise": 0.50,
        },
    )

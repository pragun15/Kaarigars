from __future__ import annotations

from rescue_env.tasks.base_task import TaskSpec


def build_easy_task() -> TaskSpec:
    return TaskSpec(
        name="sweep_and_map",
        difficulty="easy",
        description="Complete area sweep and mapping in stable conditions.",
        time_limit_minutes=90.0,
        success_criteria={
            "map_coverage": 0.95,
            "victims_detected": 0.90,
            "battery_remaining": 0.10,
        },
        environment_profile={
            "weather": "summer",
            "magnitude_range": [4.0, 5.5],
            "entrapped_rate_range": [0.05, 0.12],
            "sensor_noise": 0.0,
        },
    )

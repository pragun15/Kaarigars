from __future__ import annotations

from rescue_env.tasks.base_task import TaskSpec
from rescue_env.tasks.easy_task import build_easy_task
from rescue_env.tasks.hard_task import build_hard_task
from rescue_env.tasks.medium_task import build_medium_task


_TASK_BUILDERS = {
    "sweep_and_map": build_easy_task,
    "strategic_triage": build_medium_task,
    "extreme_rescue": build_hard_task,
}

_DIFFICULTY_TO_TASK_NAME = {
    "easy": "sweep_and_map",
    "medium": "strategic_triage",
    "hard": "extreme_rescue",
}


def get_task(task_name: str) -> TaskSpec:
    if task_name not in _TASK_BUILDERS:
        raise ValueError(f"Unknown task '{task_name}'.")
    return _TASK_BUILDERS[task_name]()


def get_task_for_difficulty(difficulty: str) -> TaskSpec:
    task_name = _DIFFICULTY_TO_TASK_NAME.get(difficulty)
    if task_name is None:
        raise ValueError(f"Unknown difficulty '{difficulty}'.")
    return get_task(task_name)


def list_tasks() -> list[TaskSpec]:
    return [builder() for builder in _TASK_BUILDERS.values()]

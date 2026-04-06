from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from rescue_env.scoring.reward_calculator import calculate_final_reward
from rescue_env.scoring.types import EpisodeStats, ScoreBreakdown, clip01
from rescue_env.tasks import get_task, get_task_for_difficulty


@dataclass
class TaskGrade:
    """Task-level grading output for agent evaluation."""

    task_name: str
    difficulty: str
    reward: float
    task_score: float
    success: bool
    criteria: dict[str, dict[str, float | bool]]
    score_breakdown: dict[str, Any]
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_task_metric_view(stats: EpisodeStats, breakdown: ScoreBreakdown) -> dict[str, float]:
    """Map raw episode stats into task-criteria metric names."""
    total_victims = max(1, stats.total_victims)
    total_critical = max(1, stats.total_critical_victims if stats.total_critical_victims > 0 else stats.total_victims)

    map_coverage = clip01(stats.map_coverage if stats.map_coverage > 0 else stats.coverage)
    victims_detected = clip01(stats.true_positives / total_victims)
    battery_remaining = clip01(stats.remaining_battery / 100.0)

    critical_victims_rescued = clip01(stats.critical_victims_rescued / total_critical)
    priority_score = clip01(breakdown.decision)
    safety_score = clip01(breakdown.safety)

    victims_rescued = clip01(stats.rescued_victims / total_victims)
    decision_score = clip01(breakdown.decision)
    mission_completion = clip01(stats.mission_completion if stats.mission_completion > 0 else breakdown.final)

    return {
        "map_coverage": map_coverage,
        "victims_detected": victims_detected,
        "battery_remaining": battery_remaining,
        "critical_victims_rescued": critical_victims_rescued,
        "priority_score": priority_score,
        "safety_score": safety_score,
        "victims_rescued": victims_rescued,
        "decision_score": decision_score,
        "mission_completion": mission_completion,
    }


def grade_episode(task_name: str, stats: EpisodeStats, breakdown: ScoreBreakdown | None = None) -> TaskGrade:
    """Grade one episode against task success criteria and final reward."""
    if breakdown is None:
        breakdown = calculate_final_reward(stats)

    task = get_task(task_name)
    metric_view = build_task_metric_view(stats, breakdown)
    task_score, success, criteria_details = task.evaluate(metric_view)

    return TaskGrade(
        task_name=task.name,
        difficulty=task.difficulty,
        reward=breakdown.final,
        task_score=task_score,
        success=success,
        criteria=criteria_details,
        score_breakdown=breakdown.to_dict(),
        metrics=metric_view,
    )


def grade_episode_by_difficulty(difficulty: str, stats: EpisodeStats, breakdown: ScoreBreakdown | None = None) -> TaskGrade:
    """Convenience wrapper to select task by difficulty."""
    task = get_task_for_difficulty(difficulty)
    return grade_episode(task.name, stats, breakdown)

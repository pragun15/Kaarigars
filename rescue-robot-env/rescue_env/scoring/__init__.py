"""Scoring package exports."""

from rescue_env.scoring.reward_calculator import calculate_final_reward, calculate_step_reward
from rescue_env.scoring.types import EpisodeStats, ScoreBreakdown


def grade_episode(*args, **kwargs):
    from rescue_env.scoring.grader import grade_episode as _grade_episode

    return _grade_episode(*args, **kwargs)


def grade_episode_by_difficulty(*args, **kwargs):
    from rescue_env.scoring.grader import grade_episode_by_difficulty as _grade_episode_by_difficulty

    return _grade_episode_by_difficulty(*args, **kwargs)

__all__ = [
    "EpisodeStats",
    "ScoreBreakdown",
    "calculate_final_reward",
    "calculate_step_reward",
    "grade_episode",
    "grade_episode_by_difficulty",
]

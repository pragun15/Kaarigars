from __future__ import annotations

from rescue_env.scoring.grader import grade_episode_by_difficulty
from rescue_env.scoring.reward_calculator import calculate_final_reward, calculate_step_reward
from rescue_env.scoring.types import EpisodeStats


def _base_stats() -> EpisodeStats:
    return EpisodeStats(
        difficulty="easy",
        total_steps=300,
        collisions=1,
        instability_events=2,
        successful_recoveries=2,
        total_victims=80,
        total_critical_victims=20,
        true_positives=70,
        false_positives=1,
        detection_confidence=0.9,
        detected_victims=72,
        accurately_located=60,
        localization_errors_m=[0.4, 1.0, 2.2, 0.6],
        rescued_victims=55,
        critical_victims_rescued=18,
        rescue_attempts=60,
        successful_rescues=55,
        total_rescues=55,
        correct_priority_rescues=50,
        correct_priority_assignments=65,
        useful_insights=50,
        total_scans=60,
        coverage=0.93,
        remaining_battery=35.0,
        work_accomplished=140.0,
        energy_used=90.0,
        revisit_ratio=0.15,
        idle_ratio=0.08,
        smoothness=0.85,
        time_elapsed_minutes=82.0,
        time_limit_minutes=90.0,
        hazards_flagged=12,
        explored_cells=800,
        effective_actions=280,
        map_coverage=0.95,
        mission_completion=0.9,
    )


def test_final_reward_in_bounds() -> None:
    breakdown = calculate_final_reward(_base_stats())
    assert 0.0 <= breakdown.final <= 1.0


def test_penalties_reduce_score() -> None:
    stats = _base_stats()
    without_penalty = calculate_final_reward(stats).final

    stats.false_explosion_trigger = 1
    stats.preventable_destruction = 1
    with_penalty = calculate_final_reward(stats).final

    assert with_penalty <= without_penalty


def test_step_reward_progress_positive() -> None:
    prev = _base_stats()
    current = _base_stats()
    current.true_positives += 2
    current.rescued_victims += 1
    current.explored_cells += 40

    reward = calculate_step_reward(prev, current, terminal=False)
    assert reward > 0.0


def test_easy_task_grading_success() -> None:
    stats = _base_stats()
    grade = grade_episode_by_difficulty("easy", stats)

    assert grade.task_name == "sweep_and_map"
    assert 0.0 <= grade.task_score <= 1.0

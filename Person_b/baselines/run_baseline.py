from __future__ import annotations

import argparse
import importlib
import random
from statistics import mean
from typing import Any

from baselines.astar_agent import AStarHeuristicAgent
from baselines.greedy_agent import GreedyTriageAgent
from baselines.random_agent import RandomAgent
from rescue_env.scoring.grader import TaskGrade, grade_episode_by_difficulty
from rescue_env.scoring.reward_calculator import calculate_final_reward
from rescue_env.scoring.types import EpisodeStats


def _difficulty_defaults(difficulty: str) -> dict[str, float]:
    if difficulty == "medium":
        return {"time_limit": 60.0, "victims": 120}
    if difficulty == "hard":
        return {"time_limit": 45.0, "victims": 220}
    return {"time_limit": 90.0, "victims": 60}


def _mock_stats(difficulty: str, seed: int) -> EpisodeStats:
    rng = random.Random(seed)
    defaults = _difficulty_defaults(difficulty)
    total_victims = int(defaults["victims"])
    total_critical = max(1, int(total_victims * rng.uniform(0.2, 0.4)))

    return EpisodeStats(
        difficulty=difficulty,
        total_steps=rng.randint(200, 700),
        collisions=rng.randint(0, 4),
        collisions_near_survivor=rng.randint(0, 1),
        joint_damage_events=rng.randint(0, 1),
        tip_over_with_recovery=rng.randint(0, 1),
        instability_events=rng.randint(0, 8),
        successful_recoveries=rng.randint(0, 8),
        total_victims=total_victims,
        total_critical_victims=total_critical,
        true_positives=rng.randint(int(total_victims * 0.4), int(total_victims * 0.95)),
        false_positives=rng.randint(0, 6),
        detection_confidence=rng.uniform(0.5, 1.0),
        detected_victims=rng.randint(int(total_victims * 0.4), int(total_victims * 0.95)),
        accurately_located=rng.randint(int(total_victims * 0.3), int(total_victims * 0.85)),
        localization_errors_m=[rng.uniform(0.2, 3.5) for _ in range(20)],
        rescued_victims=rng.randint(int(total_victims * 0.2), int(total_victims * 0.75)),
        critical_victims_rescued=rng.randint(int(total_critical * 0.2), int(total_critical * 0.95)),
        rescue_attempts=rng.randint(int(total_victims * 0.25), int(total_victims * 0.9)),
        successful_rescues=rng.randint(int(total_victims * 0.2), int(total_victims * 0.75)),
        improper_handling_events=rng.randint(0, 2),
        dropped_victims=rng.randint(0, 2),
        total_rescues=rng.randint(int(total_victims * 0.2), int(total_victims * 0.8)),
        correct_priority_rescues=rng.randint(int(total_victims * 0.1), int(total_victims * 0.7)),
        incorrect_rescue_order_events=rng.randint(0, 2),
        correct_priority_assignments=rng.randint(int(total_victims * 0.2), int(total_victims * 0.9)),
        misclassified_critical_victims=rng.randint(0, 2),
        useful_insights=rng.randint(10, 90),
        total_scans=rng.randint(20, 100),
        coverage=rng.uniform(0.4, 1.0),
        missed_major_hazards=rng.randint(0, 1),
        redundant_scans=rng.randint(0, 2),
        remaining_battery=rng.uniform(5.0, 65.0),
        work_accomplished=rng.uniform(20.0, 200.0),
        energy_used=rng.uniform(25.0, 180.0),
        revisit_ratio=rng.uniform(0.0, 0.5),
        idle_ratio=rng.uniform(0.0, 0.3),
        smoothness=rng.uniform(0.4, 1.0),
        time_elapsed_minutes=rng.uniform(defaults["time_limit"] * 0.6, defaults["time_limit"] * 1.2),
        time_limit_minutes=defaults["time_limit"],
        hazards_flagged=rng.randint(0, 20),
        explored_cells=rng.randint(50, 1200),
        effective_actions=rng.randint(30, 800),
        crush_injury_without_flag=rng.randint(0, 1),
        critical_gas_zone_entry=rng.randint(0, 1),
        false_explosion_trigger=rng.randint(0, 1),
        preventable_destruction=rng.randint(0, 1),
        map_coverage=rng.uniform(0.4, 1.0),
        mission_completion=rng.uniform(0.3, 1.0),
    )


def _load_env_class(path: str):
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _agent_from_name(name: str, seed: int):
    if name == "astar":
        return AStarHeuristicAgent()
    if name == "greedy":
        return GreedyTriageAgent()
    return RandomAgent(seed=seed)


def _run_mock(difficulty: str, episodes: int, seed: int) -> list[TaskGrade]:
    results: list[TaskGrade] = []
    for ep in range(episodes):
        stats = _mock_stats(difficulty=difficulty, seed=seed + ep)
        breakdown = calculate_final_reward(stats)
        grade = grade_episode_by_difficulty(difficulty, stats, breakdown)
        results.append(grade)
    return results


def _run_env(
    difficulty: str,
    episodes: int,
    seed: int,
    max_steps: int,
    agent_name: str,
    env_class_path: str,
) -> list[TaskGrade]:
    env_class = _load_env_class(env_class_path)
    results: list[TaskGrade] = []

    for ep in range(episodes):
        episode_seed = seed + ep
        env = env_class(difficulty=difficulty)
        agent = _agent_from_name(agent_name, seed=episode_seed)

        obs = env.reset(seed=episode_seed)
        info: dict[str, Any] = {}
        done = False
        truncated = False

        for _ in range(max_steps):
            action = agent.act(obs)
            obs, _, done, truncated, info = env.step(action)
            if done or truncated:
                break

        stats_payload = info.get("episode_stats") if isinstance(info, dict) else None
        if isinstance(stats_payload, dict):
            stats = EpisodeStats(**stats_payload)
        else:
            stats = EpisodeStats(difficulty=difficulty)

        if stats.time_limit_minutes <= 0:
            stats.time_limit_minutes = _difficulty_defaults(difficulty)["time_limit"]

        breakdown = calculate_final_reward(stats)
        grade = grade_episode_by_difficulty(difficulty, stats, breakdown)
        results.append(grade)

    return results


def _print_summary(results: list[TaskGrade]) -> None:
    rewards = [r.reward for r in results]
    task_scores = [r.task_score for r in results]
    success_rate = sum(1 for r in results if r.success) / max(1, len(results))

    print("Episodes:", len(results))
    print("Mean reward:", round(mean(rewards), 4))
    print("Mean task score:", round(mean(task_scores), 4))
    print("Success rate:", round(success_rate, 4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline evaluations for rescue environment.")
    parser.add_argument("--mode", choices=["mock", "env"], default="mock")
    parser.add_argument("--difficulty", choices=["easy", "medium", "hard"], default="easy")
    parser.add_argument("--agent", choices=["random", "greedy", "astar"], default="random")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-class", default="rescue_env.core.environment.RescueEnvironment")
    args = parser.parse_args()

    if args.mode == "mock":
        results = _run_mock(difficulty=args.difficulty, episodes=args.episodes, seed=args.seed)
    else:
        results = _run_env(
            difficulty=args.difficulty,
            episodes=args.episodes,
            seed=args.seed,
            max_steps=args.max_steps,
            agent_name=args.agent,
            env_class_path=args.env_class,
        )

    _print_summary(results)


if __name__ == "__main__":
    main()

from __future__ import annotations

from rescue_env.scoring.types import EpisodeStats, clip01


def survival_score(stats: EpisodeStats) -> float:
    """Collision and robot integrity survival component."""
    score = 1.0

    collision_penalties = [0.04, 0.08, 0.15]
    for index in range(stats.collisions):
        score -= collision_penalties[index] if index < len(collision_penalties) else collision_penalties[-1]

    score -= 0.25 * max(0, stats.collisions_near_survivor)
    score -= 0.08 * max(0, stats.joint_damage_events)
    score -= 0.10 * max(0, stats.tip_over_with_recovery)

    return clip01(score)


def stability_score(stats: EpisodeStats) -> float:
    """Terrain handling and recovery quality component."""
    total_steps = max(1, stats.total_steps)
    instability_events = max(0, stats.instability_events)

    if instability_events == 0:
        stability_recovery = 1.0
    else:
        stability_recovery = max(0.0, stats.successful_recoveries / instability_events)

    raw_score = (1.0 - (instability_events / total_steps)) * 0.7 + stability_recovery * 0.3
    return clip01(raw_score)


def safety_pillar_score(stats: EpisodeStats) -> float:
    """Safety pillar aggregate score in [0, 1]."""
    survival = survival_score(stats)
    stability = stability_score(stats)
    return clip01((survival * 0.5) + (stability * 0.5))

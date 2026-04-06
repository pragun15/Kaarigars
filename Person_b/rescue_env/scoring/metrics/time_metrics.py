from __future__ import annotations

from rescue_env.scoring.types import EpisodeStats, clip01


DEFAULT_LIMITS = {
    "easy": 90.0,
    "medium": 60.0,
    "hard": 45.0,
}


def resolve_time_limit(stats: EpisodeStats) -> float:
    if stats.time_limit_minutes > 0:
        return stats.time_limit_minutes
    return DEFAULT_LIMITS.get(stats.difficulty, 90.0)


def time_score(stats: EpisodeStats) -> float:
    """Golden-hour aware time score in [0, 1]."""
    limit = resolve_time_limit(stats)
    elapsed = max(0.0, stats.time_elapsed_minutes)

    if elapsed <= limit:
        return 1.0

    overtime = elapsed - limit
    virtual_remaining = max(0.0, limit - overtime)
    raw = (virtual_remaining / limit) * 0.6 + 0.4
    return clip01(raw)

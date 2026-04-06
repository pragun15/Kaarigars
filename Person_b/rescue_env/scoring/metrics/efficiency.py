from __future__ import annotations

from rescue_env.scoring.types import EpisodeStats, clip01


def energy_score(stats: EpisodeStats) -> float:
    battery_component = clip01(stats.remaining_battery / 100.0) * 0.5

    energy_used = max(1e-6, stats.energy_used)
    productivity_ratio = max(0.0, stats.work_accomplished / energy_used)
    productivity_component = clip01(productivity_ratio) * 0.5

    return clip01(battery_component + productivity_component)


def path_efficiency_score(stats: EpisodeStats) -> float:
    raw = 1.0 - (clip01(stats.revisit_ratio) * 0.4) - (clip01(stats.idle_ratio) * 0.3) + (clip01(stats.smoothness) * 0.3)
    return clip01(raw)


def efficiency_pillar_score(stats: EpisodeStats) -> float:
    """Efficiency pillar aggregate score in [0, 1]."""
    return clip01((energy_score(stats) * 0.5) + (path_efficiency_score(stats) * 0.5))

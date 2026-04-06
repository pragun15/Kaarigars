from __future__ import annotations

from rescue_env.scoring.types import EpisodeStats, clip01


def priority_score(stats: EpisodeStats) -> float:
    victims = max(1, stats.total_victims)
    base = max(0, stats.correct_priority_assignments) / victims
    penalties = max(0, stats.misclassified_critical_victims) * 0.15
    return clip01(base - penalties)


def environment_scanning_score(stats: EpisodeStats) -> float:
    scans = max(1, stats.total_scans)
    useful_component = (max(0, stats.useful_insights) / scans) * 0.6
    coverage_component = clip01(stats.coverage) * 0.4

    penalties = (max(0, stats.missed_major_hazards) * 0.10) + (max(0, stats.redundant_scans) * 0.05)
    return clip01(useful_component + coverage_component - penalties)


def decision_pillar_score(stats: EpisodeStats) -> float:
    """Decision pillar aggregate score in [0, 1]."""
    weighted = (priority_score(stats) * 0.12) + (environment_scanning_score(stats) * 0.08)
    return clip01(weighted / 0.20)

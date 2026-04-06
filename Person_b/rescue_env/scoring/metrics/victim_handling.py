from __future__ import annotations

from rescue_env.scoring.types import EpisodeStats, clip01


def _precision_bucket(error_m: float) -> float:
    if error_m <= 0.5:
        return 1.0
    if error_m <= 1.5:
        return 0.7
    if error_m <= 3.0:
        return 0.3
    return 0.0


def detection_score(stats: EpisodeStats) -> float:
    total_victims = max(1, stats.total_victims)
    tp_component = (max(0, stats.true_positives) / total_victims) * 0.7
    confidence_component = clip01(stats.detection_confidence) * 0.3
    fp_penalty = max(0, stats.false_positives) * 0.15
    return clip01(tp_component + confidence_component - fp_penalty)


def location_score(stats: EpisodeStats) -> float:
    detected = max(1, stats.detected_victims)
    located_component = (max(0, stats.accurately_located) / detected) * 0.6

    if stats.localization_errors_m:
        precision = sum(_precision_bucket(e) for e in stats.localization_errors_m) / len(stats.localization_errors_m)
    else:
        precision = 0.0

    precision_component = precision * 0.4
    return clip01(located_component + precision_component)


def rescue_score(stats: EpisodeStats) -> float:
    detected = max(1, stats.detected_victims)
    attempts = max(1, stats.rescue_attempts)

    rescued_component = (max(0, stats.rescued_victims) / detected) * 0.6
    success_component = (max(0, stats.successful_rescues) / attempts) * 0.4

    penalties = (max(0, stats.improper_handling_events) * 0.10) + (max(0, stats.dropped_victims) * 0.15)
    return clip01(rescued_component + success_component - penalties)


def rescue_order_score(stats: EpisodeStats) -> float:
    rescues = max(1, stats.total_rescues)
    order_component = max(0, stats.correct_priority_rescues) / rescues
    penalties = max(0, stats.incorrect_rescue_order_events) * 0.10
    return clip01(order_component - penalties)


def victim_handling_pillar_score(stats: EpisodeStats) -> float:
    """Weighted-normalized victim pillar aggregate in [0, 1]."""
    weights = {
        "detection": 0.08,
        "location": 0.10,
        "rescue": 0.10,
        "order": 0.08,
    }
    weighted = (
        detection_score(stats) * weights["detection"]
        + location_score(stats) * weights["location"]
        + rescue_score(stats) * weights["rescue"]
        + rescue_order_score(stats) * weights["order"]
    )
    return clip01(weighted / sum(weights.values()))

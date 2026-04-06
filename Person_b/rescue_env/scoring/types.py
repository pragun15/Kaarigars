from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


def clip01(value: float) -> float:
    """Clamp float values to [0, 1]."""
    return max(0.0, min(1.0, value))


@dataclass
class EpisodeStats:
    """Normalized stats required by the reward and grader pipelines."""

    difficulty: str = "easy"

    total_steps: int = 0
    collisions: int = 0
    collisions_near_survivor: int = 0
    joint_damage_events: int = 0
    tip_over_with_recovery: int = 0
    instability_events: int = 0
    successful_recoveries: int = 0

    total_victims: int = 0
    total_critical_victims: int = 0
    true_positives: int = 0
    false_positives: int = 0
    detection_confidence: float = 0.0

    detected_victims: int = 0
    accurately_located: int = 0
    localization_errors_m: list[float] = field(default_factory=list)

    rescued_victims: int = 0
    critical_victims_rescued: int = 0
    rescue_attempts: int = 0
    successful_rescues: int = 0
    improper_handling_events: int = 0
    dropped_victims: int = 0

    total_rescues: int = 0
    correct_priority_rescues: int = 0
    incorrect_rescue_order_events: int = 0

    correct_priority_assignments: int = 0
    misclassified_critical_victims: int = 0

    useful_insights: int = 0
    total_scans: int = 0
    coverage: float = 0.0
    missed_major_hazards: int = 0
    redundant_scans: int = 0

    remaining_battery: float = 100.0
    work_accomplished: float = 0.0
    energy_used: float = 1.0

    revisit_ratio: float = 0.0
    idle_ratio: float = 0.0
    smoothness: float = 1.0

    time_elapsed_minutes: float = 0.0
    time_limit_minutes: float = 90.0

    hazards_flagged: int = 0
    explored_cells: int = 0
    effective_actions: int = 0

    crush_injury_without_flag: int = 0
    critical_gas_zone_entry: int = 0
    false_explosion_trigger: int = 0
    preventable_destruction: int = 0

    map_coverage: float = 0.0
    mission_completion: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize stats to a plain dictionary for logging/debug."""
        return asdict(self)


@dataclass
class ScoreBreakdown:
    """Final and per-pillar scoring outputs."""

    safety: float
    victim_handling: float
    decision: float
    efficiency: float
    time: float
    weighted_total: float
    penalties_total: float
    final: float
    penalty_events: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize breakdown to a plain dictionary for logging/debug."""
        return asdict(self)

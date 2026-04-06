from __future__ import annotations

from rescue_env.scoring.metrics.decision import decision_pillar_score
from rescue_env.scoring.metrics.efficiency import efficiency_pillar_score
from rescue_env.scoring.metrics.safety import safety_pillar_score
from rescue_env.scoring.metrics.time_metrics import time_score
from rescue_env.scoring.metrics.victim_handling import victim_handling_pillar_score
from rescue_env.scoring.penalties import compute_absolute_penalties
from rescue_env.scoring.types import EpisodeStats, ScoreBreakdown

PILLAR_WEIGHTS = {
    "safety": 0.20,
    "victim": 0.30,
    "decision": 0.20,
    "efficiency": 0.20,
    "time": 0.10,
}


def calculate_final_reward(stats: EpisodeStats) -> ScoreBreakdown:
    """Compute final adjusted score using weighted pillars and absolute penalties."""
    safety = safety_pillar_score(stats)
    victim = victim_handling_pillar_score(stats)
    decision = decision_pillar_score(stats)
    efficiency = efficiency_pillar_score(stats)
    time_component = time_score(stats)

    weighted_total = (
        PILLAR_WEIGHTS["safety"] * safety
        + PILLAR_WEIGHTS["victim"] * victim
        + PILLAR_WEIGHTS["decision"] * decision
        + PILLAR_WEIGHTS["efficiency"] * efficiency
        + PILLAR_WEIGHTS["time"] * time_component
    )

    penalties_total, penalty_events = compute_absolute_penalties(stats)
    final_score = max(0.0, min(1.0, weighted_total - penalties_total))

    return ScoreBreakdown(
        safety=safety,
        victim_handling=victim,
        decision=decision,
        efficiency=efficiency,
        time=time_component,
        weighted_total=weighted_total,
        penalties_total=penalties_total,
        final=final_score,
        penalty_events=penalty_events,
    )


def calculate_step_reward(previous: EpisodeStats, current: EpisodeStats, terminal: bool = False) -> float:
    """Shaped step reward with optional terminal bonus from the full final score."""
    delta_tp = max(0, current.true_positives - previous.true_positives)
    delta_rescued = max(0, current.rescued_victims - previous.rescued_victims)
    delta_critical_rescued = max(0, current.critical_victims_rescued - previous.critical_victims_rescued)
    delta_hazards_flagged = max(0, current.hazards_flagged - previous.hazards_flagged)
    delta_explored_cells = max(0, current.explored_cells - previous.explored_cells)
    delta_effective_actions = max(0, current.effective_actions - previous.effective_actions)

    delta_collisions = max(0, current.collisions - previous.collisions)
    delta_dropped = max(0, current.dropped_victims - previous.dropped_victims)

    shaped = 0.0
    shaped += 0.05 * delta_tp
    shaped += 0.10 * delta_rescued
    shaped += 0.20 * delta_critical_rescued
    shaped += 0.03 * delta_hazards_flagged
    shaped += 0.002 * delta_explored_cells
    shaped += 0.001 * delta_effective_actions

    shaped -= 0.04 * delta_collisions
    shaped -= 0.15 * delta_dropped

    if terminal:
        shaped += calculate_final_reward(current).final

    return max(-1.0, min(1.0, shaped))

from __future__ import annotations

from rescue_env.scoring.types import EpisodeStats

# Absolute penalties are applied after weighted pillar aggregation.
ABSOLUTE_PENALTIES: dict[str, float] = {
    "crush_injury_without_flag": 0.25,
    "critical_gas_zone_entry": 0.15,
    "false_explosion_trigger": 0.30,
    "preventable_destruction": 0.20,
}


def compute_absolute_penalties(stats: EpisodeStats) -> tuple[float, list[dict[str, float | str]]]:
    """Compute total absolute penalties and a list of penalty events."""
    events: list[dict[str, float | str]] = []
    total = 0.0

    for penalty_name, amount in ABSOLUTE_PENALTIES.items():
        count = getattr(stats, penalty_name, 0)
        if count <= 0:
            continue

        for _ in range(count):
            events.append({"type": penalty_name, "amount": -amount})
        total += amount * count

    return total, events

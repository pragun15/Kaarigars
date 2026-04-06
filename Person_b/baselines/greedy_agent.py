from __future__ import annotations

from typing import Any


class GreedyTriageAgent:
    """Heuristic policy prioritizing critical rescues, then hazards, then exploration."""

    def act(self, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        obs = observation or {}
        nearby = obs.get("nearby", {}) if isinstance(obs, dict) else {}

        victims = nearby.get("victims", []) if isinstance(nearby, dict) else []
        hazards = nearby.get("hazards", []) if isinstance(nearby, dict) else []

        critical = [v for v in victims if isinstance(v, dict) and v.get("priority") == "critical"]
        if critical:
            victim_id = critical[0].get("id", "unknown")
            return {
                "action_type": "rescue_victim",
                "parameters": {
                    "victim_id": victim_id,
                    "handling_method": "gentle",
                },
            }

        if hazards:
            h0 = hazards[0] if isinstance(hazards[0], dict) else {}
            return {
                "action_type": "flag_hazard",
                "parameters": {
                    "hazard_type": h0.get("type", "structural"),
                    "location": h0.get("location", [0.0, 0.0]),
                },
            }

        robot_status = obs.get("robot_status", {}) if isinstance(obs, dict) else {}
        battery_level = float(robot_status.get("battery_level", 100.0)) if isinstance(robot_status, dict) else 100.0
        if battery_level < 5.0:
            return {"action_type": "idle", "parameters": {}}

        if victims:
            target = victims[0].get("position", [0.0, 0.0]) if isinstance(victims[0], dict) else [0.0, 0.0]
            return {
                "action_type": "move",
                "parameters": {
                    "target_position": target[:2],
                    "speed": 0.8,
                },
            }

        return {
            "action_type": "scan_thermal",
            "parameters": {
                "duration": 1.0,
            },
        }

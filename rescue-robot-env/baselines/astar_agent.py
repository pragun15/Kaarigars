from __future__ import annotations

import heapq
from typing import Any


class AStarHeuristicAgent:
    """A lightweight A* style planner over a local occupancy grid if available."""

    def act(self, observation: dict[str, Any] | None = None) -> dict[str, Any]:
        obs = observation or {}
        nearby = obs.get("nearby", {}) if isinstance(obs, dict) else {}
        victims = nearby.get("victims", []) if isinstance(nearby, dict) else []

        robot_status = obs.get("robot_status", {}) if isinstance(obs, dict) else {}
        position = robot_status.get("position", [0.0, 0.0, 0.0]) if isinstance(robot_status, dict) else [0.0, 0.0, 0.0]
        start = (int(position[0]), int(position[1]))

        if victims:
            target_data = victims[0] if isinstance(victims[0], dict) else {}
            target_pos = target_data.get("position", [0.0, 0.0])
            goal = (int(target_pos[0]), int(target_pos[1]))

            local_grid = obs.get("local_grid")
            next_step = self._next_step(start=start, goal=goal, local_grid=local_grid)
            if next_step is not None:
                return {
                    "action_type": "move",
                    "parameters": {
                        "target_position": [float(next_step[0]), float(next_step[1])],
                        "speed": 1.0,
                    },
                }

            victim_id = target_data.get("id", "unknown")
            return {
                "action_type": "rescue_victim",
                "parameters": {
                    "victim_id": victim_id,
                    "handling_method": "gentle",
                },
            }

        return {
            "action_type": "scan_lidar",
            "parameters": {
                "duration": 1.0,
            },
        }

    def _next_step(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        local_grid: list[list[int]] | None,
    ) -> tuple[int, int] | None:
        if start == goal:
            return start

        # Fallback to direct motion when grid is unavailable.
        if not local_grid:
            dx = 1 if goal[0] > start[0] else -1 if goal[0] < start[0] else 0
            dy = 1 if goal[1] > start[1] else -1 if goal[1] < start[1] else 0
            return start[0] + dx, start[1] + dy

        rows = len(local_grid)
        cols = len(local_grid[0]) if rows else 0

        def in_bounds(cell: tuple[int, int]) -> bool:
            return 0 <= cell[0] < rows and 0 <= cell[1] < cols

        def passable(cell: tuple[int, int]) -> bool:
            return local_grid[cell[0]][cell[1]] == 0

        def heuristic(a: tuple[int, int], b: tuple[int, int]) -> int:
            return abs(a[0] - b[0]) + abs(a[1] - b[1])

        frontier: list[tuple[int, tuple[int, int]]] = []
        heapq.heappush(frontier, (0, start))
        came_from: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        cost_so_far: dict[tuple[int, int], int] = {start: 0}

        neighbors = [(1, 0), (-1, 0), (0, 1), (0, -1)]

        while frontier:
            _, current = heapq.heappop(frontier)
            if current == goal:
                break

            for dx, dy in neighbors:
                nxt = (current[0] + dx, current[1] + dy)
                if not in_bounds(nxt) or not passable(nxt):
                    continue

                new_cost = cost_so_far[current] + 1
                if nxt not in cost_so_far or new_cost < cost_so_far[nxt]:
                    cost_so_far[nxt] = new_cost
                    priority = new_cost + heuristic(nxt, goal)
                    heapq.heappush(frontier, (priority, nxt))
                    came_from[nxt] = current

        if goal not in came_from:
            return None

        path = [goal]
        while path[-1] != start:
            parent = came_from[path[-1]]
            if parent is None:
                break
            path.append(parent)

        path.reverse()
        if len(path) <= 1:
            return start
        return path[1]

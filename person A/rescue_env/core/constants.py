"""Constants and default profiles for the rescue environment."""

DIFFICULTY_PROFILES = {
    "easy": {
        "map_size": 32,
        "victim_range": (40, 80),
        "time_limit_minutes": 90.0,
        "battery_capacity": 100.0,
        "sensor_noise": 0.0,
        "battery": {"move": 0.8, "sensor": 0.3, "idle": 0.1},
        "hazard_density": 0.01,
        "debris_density": 0.08,
    },
    "medium": {
        "map_size": 40,
        "victim_range": (100, 150),
        "time_limit_minutes": 60.0,
        "battery_capacity": 100.0,
        "sensor_noise": 0.2,
        "battery": {"move": 1.2, "sensor": 0.5, "idle": 0.2},
        "hazard_density": 0.03,
        "debris_density": 0.18,
    },
    "hard": {
        "map_size": 48,
        "victim_range": (150, 300),
        "time_limit_minutes": 45.0,
        "battery_capacity": 100.0,
        "sensor_noise": 0.5,
        "battery": {"move": 2.5, "sensor": 0.8, "idle": 0.5},
        "hazard_density": 0.06,
        "debris_density": 0.3,
    },
}

MAX_SPEED_MS = 1.6
MIN_SPEED_MS = 0.8
GRID_RESOLUTION_METERS = 1.0

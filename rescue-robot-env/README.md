# Rescue Robot OpenEnv

Production-focused OpenEnv environment for earthquake search and rescue.

## Current Status
This folder is the merged project containing:
1. Environment simulation core (reset/step/state API).
2. Multi-pillar scoring and penalties.
3. Easy/medium/hard task grading.
4. Baseline agents and evaluation runner.

## Quick Start
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run scoring unit tests:
   - `pytest -q tests/test_scoring.py`
3. Run baseline in mock mode:
   - `python -m baselines.run_baseline --mode mock --episodes 5 --seed 42`
4. Run baseline in environment mode:
   - `python -m baselines.run_baseline --mode env --difficulty easy --episodes 3 --seed 42`

## Repository Areas
- `rescue_env/core/`: OpenEnv-compatible environment shell and config.
- `rescue_env/models/`: typed action, observation, state, robot, victim models.
- `rescue_env/world/`: map, debris, hazards, physics, victim generation.
- `rescue_env/robot/`: controller, battery, sensor simulation hooks.
- `rescue_env/scoring/`: reward pillars, penalties, grader.
- `rescue_env/tasks/`: easy/medium/hard task definitions and criteria.
- `baselines/`: baseline policies and runner.
- `openenv.yaml`: OpenEnv spec entry point.

## Verification Commands
1. `pytest -q tests/test_scoring.py`
2. `python -m baselines.run_baseline --mode mock --difficulty easy --episodes 5 --seed 42`
3. `python -m baselines.run_baseline --mode env --difficulty easy --episodes 2 --seed 42`

## Notes
The environment now emits `episode_stats` and `score_breakdown` in step `info` payloads for grading and baseline evaluation.

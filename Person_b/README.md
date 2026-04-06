# Rescue Robot OpenEnv

Production-focused OpenEnv environment for earthquake search and rescue.

## Current Status
This repository is under active implementation in two parallel streams:
1. Person A: environment simulation core.
2. Person B: scoring, tasks, baseline evaluation, and packaging.

## Quick Start
1. Install dependencies:
   - `pip install -r requirements.txt`
2. Run scoring unit tests:
   - `pytest -q tests/test_scoring.py`
3. Run baseline in mock mode (available now):
   - `python -m baselines.run_baseline --mode mock --episodes 5 --seed 42`

## Repository Areas
- `rescue_env/scoring/`: reward pillars, penalties, grader.
- `rescue_env/tasks/`: easy/medium/hard task definitions and criteria.
- `baselines/`: baseline policies and runner.
- `openenv.yaml`: OpenEnv spec entry point.

## Integration Contract (with environment core)
The runtime environment should provide metric counters compatible with `EpisodeStats` in `rescue_env/scoring/types.py`.

## Next Milestones
1. Integrate scoring pipeline into live `env.step`.
2. Validate final reward and task grade behavior on all difficulties.
3. Publish baseline reproducibility table in this README.

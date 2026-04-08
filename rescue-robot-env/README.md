---
title: Rescue Robot OpenEnv
sdk: docker
app_port: 7860
tags:
  - openenv
---

# Rescue Robot OpenEnv

Earthquake search-and-rescue simulation environment designed for training and evaluating agentic policies on a real-world utility task.

## Why this environment

Urban search and rescue requires balancing speed, safety, triage priority, and constrained battery/time resources.
This environment models those trade-offs in a deterministic, programmatically graded setup suitable for reproducible benchmarking.

## OpenEnv Compliance Summary

- Typed models:
  - `rescue_env/models/actions.py`
  - `rescue_env/models/observations.py`
  - `rescue_env/models/state.py`
- Environment API:
  - `reset(seed) -> Observation`
  - `step(action) -> (Observation, reward, done, truncated, info)`
  - `state() -> StateSnapshot`
- OpenEnv metadata: `openenv.yaml`
- Tasks with deterministic graders (easy/medium/hard): implemented in `rescue_env/tasks/` and `rescue_env/scoring/grader.py`
- Root baseline inference script required by submission rules: `inference.py`

## Action Space

Action payload format used by the environment:

```json
{
  "action_type": "move|scan_lidar|scan_thermal|scan_gas|listen|rescue_victim|flag_hazard|idle",
  "parameters": {}
}
```

Action details:

1. `move`
   - parameters: `target_position: [x, y]`, `speed: float`
2. `scan_lidar` / `scan_thermal` / `scan_gas` / `listen`
   - parameters: `{}`
3. `rescue_victim`
   - parameters: `victim_id: str`, `handling_method: str`
4. `flag_hazard`
   - parameters: `hazard_type: str`, `location: [x, y]`
5. `idle`
   - parameters: `{}`

## Observation Space

Primary top-level fields returned by `reset()` and `step()`:

1. `robot_status`
   - `position`, `orientation_deg`, `battery_level`, `is_stable`, `carrying_victim`
2. `sensors`
   - `lidar_points`, `thermal_signatures`, `gas_levels`, `acoustic_events`
3. `nearby_victims`
4. `nearby_hazards`
5. `time_remaining`
6. `mission_progress`

## Task Suite and Graders

Three deterministic tasks are included, each scored in `[0.0, 1.0]`.

1. Easy: `sweep_and_map`
   - Goal: high map coverage and detection in stable conditions.
   - Criteria: `map_coverage`, `victims_detected`, `battery_remaining`.
2. Medium: `strategic_triage`
   - Goal: prioritize critical rescues with safety under degraded sensing.
   - Criteria: `critical_victims_rescued`, `priority_score`, `safety_score`.
3. Hard: `extreme_rescue`
   - Goal: mass-casualty recovery under severe hazard pressure.
   - Criteria: `victims_rescued`, `decision_score`, `mission_completion`.

Grading and score normalization logic:

- `rescue_env/tasks/base_task.py`
- `rescue_env/scoring/grader.py`

## Reward Design

The reward is shaped over the full trajectory and includes:

1. Positive progress signals
   - new victim detection, rescue events, critical rescue events, hazard flagging, exploration, effective actions.
2. Negative behavior penalties
   - collisions, dropped victims, unsafe or destructive behavior penalties.
3. Terminal score component
   - weighted multi-pillar final score with absolute penalties.

Reward and final-score modules:

- `rescue_env/scoring/reward_calculator.py`
- `rescue_env/scoring/penalties.py`
- `rescue_env/scoring/metrics/`

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure OpenEnv CLI is available:

```bash
pip install openenv-core
openenv --version
```

4. Run tests:

```bash
pytest -q tests/test_scoring.py tests/test_env_integration.py
```

## Required Inference Environment Variables

The root `inference.py` consumes the required variables from submission instructions:

1. `API_BASE_URL`
2. `MODEL_NAME`
3. `HF_TOKEN`

Optional compatibility aliases:

4. `OPENAI_API_KEY`
5. `API_KEY`

Copy `.env.example` to `.env` and set values.

## Baseline Inference (Mandatory Script)

Run:

```bash
python inference.py
```

The script:

1. Uses OpenAI client SDK (`from openai import OpenAI`).
2. Runs all 3 tasks by default.
3. Emits strict structured stdout lines:
   - `[START] task=... env=... model=...`
   - `[STEP] step=... action=... reward=... done=... error=...`
   - `[END] success=... steps=... score=... rewards=...`

## Reproducible Baseline Snapshot

Local baseline run (`python -m baselines.run_baseline --mode env --episodes 5 --seed 42`) produced:

1. Easy
   - Mean reward: `0.5432`
   - Mean task score: `0.5474`
   - Success rate: `0.0`
2. Medium
   - Mean reward: `0.5275`
   - Mean task score: `0.3702`
   - Success rate: `0.0`
3. Hard
   - Mean reward: `0.4944`
   - Mean task score: `0.2907`
   - Success rate: `0.0`

These are non-trained heuristic baselines and are intended as reproducible reference points.

## HF Space Deployment

Container serves a persistent API on port `7860` using `uvicorn app:app`.

Endpoints:

1. `GET /` ping
2. `GET /health` health check
3. `POST /reset` reset environment
4. `POST /step` step environment
5. `GET /state` current full state

Local smoke test:

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 7860
```

## Docker

Build:

```bash
docker build -t rescue-robot-openenv .
```

Run:

```bash
docker run --rm -p 7860:7860 rescue-robot-openenv
```

## Pre-Submission Validation

Use helper script:

```bash
bash scripts/validate-submission.sh <your_space_url> .
```

It checks:

1. Required files exist (`openenv.yaml`, `Dockerfile`, root `inference.py`).
2. `openenv validate` (required; install with `pip install openenv-core`).
3. `docker build` (if Docker installed).
4. Space `POST /reset` response (`200`).
5. Space root endpoint response (`200`).

## Repository Layout

1. `rescue_env/` core environment, world simulation, typed models, scoring, tasks.
2. `baselines/` random/greedy/astar baseline agents and runner.
3. `tests/` scoring and environment integration tests.
4. `openenv.yaml` environment metadata.
5. `inference.py` mandatory OpenAI-client baseline inference script.
6. `app.py` live API service for HF deployment.

# Person B Execution Plan (Scoring, Tasks, Baselines, Packaging)

## Goal
Ship a complete evaluation and delivery lane that can integrate with Person A's environment core with minimal merge risk.

## Working Contract with Person A (freeze this first)
1. `env.step(action)` returns `(observation, reward, done, truncated, info)`.
2. `info` must contain stable metric keys consumed by scoring and grading modules.
3. `env.state()` returns a snapshot containing episode counters needed for final grading.
4. Difficulty values are exactly: `easy`, `medium`, `hard`.

## Parallel Plan by Phase

### Phase B1 (Day 1): Define scoring interfaces and deterministic grading
- Implement scoring types and metric calculators.
- Implement absolute penalties and final clipping.
- Implement task criteria evaluators and pass/fail grading.
- Deliverable: scoring package testable from synthetic stats only.

### Phase B2 (Day 2): Integrate task definitions and evaluation pipeline
- Implement easy/medium/hard task definitions.
- Implement task-aware time limits and success criteria.
- Add a unified grader API to evaluate episodes and produce agent-friendly feedback.
- Deliverable: one function call to compute score and task success.

### Phase B3 (Day 3): Baseline framework
- Implement random and greedy baseline agents.
- Implement baseline runner with fixed seeds for reproducible runs.
- Add mock execution mode until Person A env is merged.
- Deliverable: stable baseline score table output.

### Phase B4 (Day 3-4): Packaging and docs
- Draft `openenv.yaml` with tasks, spaces, and metadata.
- Add `requirements.txt` and README skeleton with run instructions.
- Deliverable: repository is structured for OpenEnv and CI validation.

### Phase B5 (Integration day): Harden with Person A
- Map Person A's runtime metrics to scoring input schema.
- Run 10 seeded episodes per difficulty for each baseline.
- Record reference scores and lock in regression thresholds.
- Deliverable: reproducible baseline report and validated reward behavior.

## Exact Task Split for You (Person B)
1. Own all files under `rescue_env/scoring/`.
2. Own all files under `rescue_env/tasks/`.
3. Own files under `baselines/` and top-level packaging docs (`openenv.yaml`, `README.md`, `requirements.txt`).
4. Own `tests/test_scoring.py` and grading tests.

## Daily Checklist
1. Morning: run scoring unit tests and baseline mock runs.
2. Midday: sync interface changes with Person A (`info` keys, state schema).
3. End of day: update baseline score snapshots and unresolved dependency list.

## Blockers to Watch
1. Missing metric fields in `info` payload from Person A.
2. Undefined observation schema for baseline greedy policy.
3. Time/battery termination logic drift between core env and grader.

## Definition of Done for Person B
1. Final reward and penalty engine is deterministic and tested.
2. All three tasks have explicit success criteria and grader outputs.
3. Baseline script runs with fixed seeds and prints per-difficulty mean score.
4. `openenv.yaml`, `Dockerfile` hook points, and README setup path are ready for deploy lane.

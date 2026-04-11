# Kaarigars - OpenEnv Search and Rescue

Professional project repository for Team Kaarigars' OpenEnv Round 1 submission.

## Overview

Kaarigars models a real-world earthquake search-and-rescue scenario as an OpenEnv-compatible environment. The project is designed for agent training and evaluation with typed interfaces, deterministic grading, progressive task difficulty, and deployment-ready infrastructure.

## Repository Layout

This repository uses a mono-repo style layout:

- Kaarigars/ (main repository)
- Kaarigars/rescue-robot-env/ (submission project root)

Submission-critical files are located at the root of Kaarigars/rescue-robot-env.

## Key Contents

- Kaarigars/rescue-robot-env - Complete OpenEnv environment project
- Kaarigars/Workflow.md - Architecture and implementation documentation
- Kaarigars/CONTEXT_DOCUMENT.txt - Project context notes

## Submission Project (rescue-robot-env)

Primary deliverables in the submission root:

- openenv.yaml
- inference.py
- Dockerfile
- README.md
- pyproject.toml

Core package and supporting modules:

- rescue_env/ (environment core, models, tasks, scoring, world simulation)
- tests/ (integration and scoring tests)
- scripts/ (local checks and pre-submission validator)

## Local Setup

Run all commands from Kaarigars/rescue-robot-env.

```bash
pip install -r requirements.txt
```

## Quality and Compliance Checks

```bash
openenv validate
pytest -q tests/test_scoring.py tests/test_env_integration.py
python scripts/pre_submit_local_check.py
```

## Deployment Validation

```bash
bash scripts/validate-submission.sh <space_runtime_url> .
```

Runtime URL format:

```text
https://<username>-<space-name>.hf.space
```

## Notes for Submission

- Keep Kaarigars/rescue-robot-env as the submission root.
- Keep required submission files at that root level.
- Final platform submission is performed by the team lead account.

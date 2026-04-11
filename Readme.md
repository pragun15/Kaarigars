# Kaarigars OpenEnv Submission Repository

This repository contains Team Kaarigars assets for the OpenEnv Round 1 hackathon.

## Submission Root

The actual environment project root is:

- `rescue-robot-env/`

Do not submit from the parent folder. All required submission files are intentionally at `rescue-robot-env/` root.

## What Is Inside

- `rescue-robot-env/` - final OpenEnv environment project
- `Workflow.md` - architecture and implementation notes
- `CONTEXT_DOCUMENT.txt` - planning/context notes

## Required Files (Inside `rescue-robot-env/`)

- `openenv.yaml`
- `inference.py`
- `Dockerfile`
- `README.md`
- `pyproject.toml`

## Quick Validation

From `rescue-robot-env/`:

```bash
openenv validate
pytest -q tests/test_scoring.py tests/test_env_integration.py
python scripts/pre_submit_local_check.py
```

## Full Pre-Submission Validator

```bash
bash scripts/validate-submission.sh <space_runtime_url> .
```

Example runtime URL format:

```text
https://<username>-<space-name>.hf.space
```

## Final Submission Notes

- Keep current file structure unchanged.
- Keep required files at `rescue-robot-env/` root.
- Final platform submission should be done by the team lead account.

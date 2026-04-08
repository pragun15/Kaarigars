#!/usr/bin/env bash
set -uo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PING_URL="${1:-}"
REPO_DIR="${2:-.}"
IMAGE_NAME="rescue-robot-openenv:local"

if [[ -z "${PING_URL}" ]]; then
  echo -e "${YELLOW}[WARN] No ping URL provided; running in local-only mode.${NC}"
fi

if [[ "${LOCAL_CHECK_STRICT_AUTH:-0}" == "1" ]]; then
  echo -e "${YELLOW}[WARN] LOCAL_CHECK_STRICT_AUTH=1 enabled; auth errors in inference output will fail local check.${NC}"
fi

pass() { echo -e "${GREEN}[PASS] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
fail() { echo -e "${RED}[FAIL] $1${NC}"; exit 1; }

cd "${REPO_DIR}" || fail "Cannot cd into repo_dir=${REPO_DIR}"

[[ -f "openenv.yaml" ]] || fail "openenv.yaml missing"
[[ -f "Dockerfile" ]] || fail "Dockerfile missing"
[[ -f "inference.py" ]] || fail "inference.py missing at repository root"
[[ -f "scripts/pre_submit_local_check.py" ]] || fail "scripts/pre_submit_local_check.py missing"
pass "Required files exist"

if python scripts/pre_submit_local_check.py; then
  pass "local pre-submit validator passed"
else
  fail "local pre-submit validator failed"
fi

if command -v openenv >/dev/null 2>&1; then
  if openenv validate; then
    pass "openenv validate passed"
  else
    fail "openenv validate failed"
  fi
else
  fail "openenv command not found (install with: pip install openenv-core)"
fi

if command -v docker >/dev/null 2>&1; then
  if docker build -t "${IMAGE_NAME}" .; then
    pass "docker build passed"
  else
    fail "docker build failed"
  fi
else
  warn "docker not installed; skipping docker build"
fi

if [[ -n "${PING_URL}" ]]; then
  if curl -fsS -X POST "${PING_URL}/reset" \
    -H 'Content-Type: application/json' \
    -d '{}' >/dev/null; then
    pass "Space /reset responded 200"
  else
    fail "Space /reset did not respond with success"
  fi

  if curl -fsS "${PING_URL}" >/dev/null; then
    pass "Space root endpoint responded 200"
  else
    fail "Space root endpoint did not respond with success"
  fi
else
  warn "Skipping remote Space ping/reset checks in local-only mode"
fi

if [[ -n "${HF_TOKEN:-}" && -n "${API_BASE_URL:-}" && -n "${MODEL_NAME:-}" ]]; then
  if python inference.py >/dev/null; then
    pass "inference.py completed"
  else
    fail "inference.py failed"
  fi
else
  warn "HF_TOKEN/API_BASE_URL/MODEL_NAME not all set; skipped live inference run"
fi

echo -e "${GREEN}Validation script completed.${NC}"

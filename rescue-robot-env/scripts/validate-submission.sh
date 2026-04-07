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
  echo -e "${RED}Usage: ./scripts/validate-submission.sh <ping_url> [repo_dir]${NC}"
  exit 2
fi

pass() { echo -e "${GREEN}[PASS] $1${NC}"; }
warn() { echo -e "${YELLOW}[WARN] $1${NC}"; }
fail() { echo -e "${RED}[FAIL] $1${NC}"; exit 1; }

cd "${REPO_DIR}" || fail "Cannot cd into repo_dir=${REPO_DIR}"

[[ -f "openenv.yaml" ]] || fail "openenv.yaml missing"
[[ -f "Dockerfile" ]] || fail "Dockerfile missing"
[[ -f "inference.py" ]] || fail "inference.py missing at repository root"
pass "Required files exist"

if command -v openenv >/dev/null 2>&1; then
  if openenv validate; then
    pass "openenv validate passed"
  else
    fail "openenv validate failed"
  fi
else
  warn "openenv CLI not installed; skipping openenv validate"
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

if curl -fsS "${PING_URL}" >/dev/null; then
  pass "Space ping URL responded 200"
else
  fail "Space ping URL did not respond with success"
fi

if curl -fsS -X POST "${PING_URL}/reset" \
  -H 'Content-Type: application/json' \
  -d '{"difficulty":"easy","seed":42}' >/dev/null; then
  pass "Space reset endpoint responded"
else
  fail "Space reset endpoint failed"
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

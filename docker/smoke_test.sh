#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — Validate a locally built m3_environ Docker image
#
# Usage (from project root):
#   bash docker/smoke_test.sh [image_name]
#
# Defaults:
#   image_name = m3_environ
#
# What it checks:
#   1. Required files exist inside the image
#   2. M3 REST FastAPI starts and serves /openapi.json (port 8000)
#   3. BPO MCP server responds to the MCP initialize handshake
#   4. M3 REST MCP server responds to the MCP initialize handshake
#
# Exit code: 0 = all checks passed, 1 = one or more checks failed
# =============================================================================
set -euo pipefail

IMAGE="${1:-m3_environ}"
CONTAINER="m3_smoke_$$"
PASS=0
FAIL=0

# Resolve project root (directory containing this script's parent)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DB_DIR="$PROJECT_ROOT/data/db"
CONFIGS_DIR="$PROJECT_ROOT/apis/configs"

# Colours (disabled if not a terminal)
if [ -t 1 ]; then
  GREEN="\033[0;32m"; RED="\033[0;31m"; CYAN="\033[0;36m"; RESET="\033[0m"
else
  GREEN=""; RED=""; CYAN=""; RESET=""
fi

MCP_INIT='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2024-11-05","capabilities":{},"clientInfo":{"name":"smoke-test","version":"0.1.0"}}}'

pass() { echo -e "${GREEN}  [PASS]${RESET} $*"; ((PASS++)) || true; }
fail() { echo -e "${RED}  [FAIL]${RESET} $*"; ((FAIL++)) || true; }
section() { echo -e "\n${CYAN}=== $* ===${RESET}"; }

cleanup() {
  docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# ---------------------------------------------------------------------------
section "1. Image file checks"
# ---------------------------------------------------------------------------

check_file() {
  local path="$1"
  if docker run --rm --entrypoint /bin/sh "$IMAGE" -c "test -e '$path'" 2>/dev/null; then
    pass "$path"
  else
    fail "$path  (not found in image)"
  fi
}

check_file /app/m3-rest/app.py
check_file /app/m3-rest/mcp_server.py
check_file /app/retrievers/server.py
check_file /app/retrievers/mcp_server.py
check_file /app/apis/__init__.py
check_file /app/apis/bpo/__init__.py
check_file /app/apis/bpo/mcp/server.py
check_file /app/apis/bpo/api/schemas.py
check_file /app/apis/bpo/api/candidate_source.py
check_file /app/apis/bpo/api/skills.py
check_file /app/apis/bpo/data/candidate_data.parquet
check_file /app/entrypoint.sh

# ---------------------------------------------------------------------------
section "2. M3 REST FastAPI health (port 8000)"
# ---------------------------------------------------------------------------

if [ ! -d "$DB_DIR" ]; then
  echo "  WARNING: $DB_DIR not found — skipping FastAPI health check (run 'make download' first)"
  SKIP_FASTAPI=true
else
  SKIP_FASTAPI=false
  echo "  Starting container ${CONTAINER} ..."
  docker run -d --name "$CONTAINER" \
    -e MCP_DB_ROOT=/app/db \
    -v "$DB_DIR:/app/db:ro" \
    -v "$CONFIGS_DIR:/app/apis/configs:ro" \
    "$IMAGE" >/dev/null
fi

if [ "$SKIP_FASTAPI" = "false" ]; then
  echo "  Waiting up to 60s for M3 REST to be ready ..."
  READY=false
  for i in $(seq 1 60); do
    if docker exec "$CONTAINER" curl -sf http://localhost:8000/openapi.json >/dev/null 2>&1; then
      READY=true
      break
    fi
    sleep 1
  done

  if [ "$READY" = "true" ]; then
    pass "M3 REST /openapi.json responded"
    # Extra: confirm at least one route is present
    ROUTE_COUNT=$(docker exec "$CONTAINER" \
      curl -sf http://localhost:8000/openapi.json \
      | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('paths',{})))" 2>/dev/null || echo 0)
    if [ "$ROUTE_COUNT" -gt 0 ] 2>/dev/null; then
      pass "M3 REST OpenAPI spec has ${ROUTE_COUNT} route(s)"
    else
      fail "M3 REST OpenAPI spec has 0 routes (unexpected)"
    fi
  else
    fail "M3 REST did not start within 60 seconds"
  fi
fi

# ---------------------------------------------------------------------------
section "3. BPO MCP server (stdio handshake)"
# Runs python directly (--entrypoint bypass) — no db or FastAPI required.
# ---------------------------------------------------------------------------

BPO_RESPONSE=$(echo "$MCP_INIT" \
  | docker run --rm -i --entrypoint python \
    "$IMAGE" /app/apis/bpo/mcp/server.py 2>/dev/null \
  | head -n 1 || true)

if echo "$BPO_RESPONSE" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); assert 'result' in d or 'id' in d" 2>/dev/null; then
  pass "BPO MCP server returned a valid JSON-RPC response"
else
  fail "BPO MCP server did not return a valid JSON-RPC response"
  echo "  Raw response: ${BPO_RESPONSE:0:200}"
fi

# ---------------------------------------------------------------------------
section "4. M3 REST MCP server (stdio handshake)"
# Requires the FastAPI server to be running (it calls localhost:8000/openapi.json
# at startup). Uses docker exec against the container from section 2.
# Skipped when data/db is not available (same condition as section 2).
# ---------------------------------------------------------------------------

if [ "$SKIP_FASTAPI" = "true" ]; then
  echo "  Skipping — FastAPI server not running (no data/db). Run 'make download' first."
else
  M3_MCP_RESPONSE=$(echo "$MCP_INIT" \
    | docker exec -i -e MCP_DOMAIN=superhero "$CONTAINER" \
      python /app/m3-rest/mcp_server.py 2>/dev/null \
    | head -n 1 || true)

  if echo "$M3_MCP_RESPONSE" | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); assert 'result' in d or 'id' in d" 2>/dev/null; then
    pass "M3 REST MCP server returned a valid JSON-RPC response"
  else
    fail "M3 REST MCP server did not return a valid JSON-RPC response"
    echo "  Raw response: ${M3_MCP_RESPONSE:0:200}"
  fi
fi

# ---------------------------------------------------------------------------
section "Summary"
# ---------------------------------------------------------------------------

TOTAL=$((PASS + FAIL))
echo ""
echo "  Passed: ${PASS}/${TOTAL}"
echo "  Failed: ${FAIL}/${TOTAL}"
echo ""

if [ "$FAIL" -eq 0 ]; then
  echo -e "${GREEN}All checks passed. Image looks good to push.${RESET}"
  exit 0
else
  echo -e "${RED}${FAIL} check(s) failed. Fix the issues before pushing.${RESET}"
  exit 1
fi

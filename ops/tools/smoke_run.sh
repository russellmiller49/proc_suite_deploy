#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-}"
if [[ -z "${PORT}" ]]; then
  PORT="$(python - <<'PY'
import socket
s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
)"
fi

BASE_URL="http://${HOST}:${PORT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
    wait "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

# Make the smoke test safe by default (no external LLM calls, no PHI review gating).
export PROCSUITE_SKIP_DOTENV="${PROCSUITE_SKIP_DOTENV:-1}"
export SKIP_WARMUP="${SKIP_WARMUP:-1}"
export CODER_REQUIRE_PHI_REVIEW="${CODER_REQUIRE_PHI_REVIEW:-false}"
export GEMINI_OFFLINE="${GEMINI_OFFLINE:-1}"
export REGISTRY_USE_STUB_LLM="${REGISTRY_USE_STUB_LLM:-1}"
export OPENAI_OFFLINE="${OPENAI_OFFLINE:-1}"
export DISABLE_STATIC_FILES="${DISABLE_STATIC_FILES:-true}"
export PORT
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"

echo "[smoke] Starting uvicorn on ${BASE_URL}"

UVICORN=( )
if python -c "import uvicorn" >/dev/null 2>&1; then
  UVICORN=(python -m uvicorn)
elif command -v uvicorn >/dev/null 2>&1; then
  UVICORN=(uvicorn)
else
  echo "[smoke] ERROR: uvicorn is not installed in the active Python environment."
  exit 1
fi

"${UVICORN[@]}" app.api.fastapi_app:app \
  --host "${HOST}" \
  --port "${PORT}" \
  --log-level warning &
SERVER_PID="$!"

echo "[smoke] Waiting for /health..."
for _ in $(seq 1 40); do
  if curl -fsS "${BASE_URL}/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.25
done
curl -fsS "${BASE_URL}/health" >/dev/null

echo "[smoke] Checking /ready..."
curl -fsS "${BASE_URL}/ready" >/dev/null

echo "[smoke] Posting minimal coder request (rules_only)..."
curl -fsS -X POST "${BASE_URL}/v1/coder/run?mode=rules_only" \
  -H "Content-Type: application/json" \
  -d '{"note":"Synthetic note: Bronchoscopy performed. Airways inspected. No biopsy performed.","locality":"00"}' \
  >/dev/null

echo "[smoke] OK"

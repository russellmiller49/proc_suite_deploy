#!/usr/bin/env bash
# Optional Railway start command using Gunicorn prefork + preload.
#
# WARNING:
# - Gunicorn is not included by default in this repo; install it before using.
# - Prefork workers increase memory usage; use only on higher-RAM plans.
# - Avoid starting background threads *before* prefork.
#
# Suggested Railway Start Command (optional):
#   ops/railway_start_gunicorn.sh

set -euo pipefail

if ! command -v gunicorn >/dev/null 2>&1; then
  echo "[railway_start_gunicorn] ERROR: gunicorn not installed in this environment."
  echo "[railway_start_gunicorn] Install gunicorn and retry."
  exit 1
fi

export PYTHONPATH="${PYTHONPATH:-}:${PWD}"
echo "[railway_start_gunicorn] PYTHONPATH=${PYTHONPATH}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

PORT="${PORT:-8000}"
WORKERS="${WORKERS:-2}"
TIMEOUT="${TIMEOUT:-120}"

echo "[railway_start_gunicorn] PORT=${PORT}"
echo "[railway_start_gunicorn] WORKERS=${WORKERS}"
echo "[railway_start_gunicorn] TIMEOUT=${TIMEOUT}"

# Optional: run Alembic migrations on start (recommended for single-instance Railway deploys).
if [[ "${PROCSUITE_RUN_MIGRATIONS_ON_START:-}" =~ ^(1|true|yes)$ ]]; then
  echo "[railway_start_gunicorn] Running migrations (alembic upgrade head)..."
  alembic upgrade head
fi

# Optional: bootstrap granular NER ONNX bundle from S3 at container start.
# If configured, this must succeed; otherwise the service should fail fast.
if [[ -n "${GRANULAR_NER_BUNDLE_S3_URI_ONNX:-${GRANULAR_NER_BUNDLE_S3_URI:-}}" ]]; then
  echo "[railway_start_gunicorn] Bootstrapping granular NER bundle from S3..."
  python ops/tools/bootstrap_granular_ner_bundle.py
fi

# Optional: bootstrap PHI redactor vendor bundle from S3 before app starts.
if [[ -n "${PHI_REDACTOR_VENDOR_BUNDLE_S3_URI:-${PHI_REDACTOR_VENDOR_BUNDLE_S3_URI_ONNX:-}}" ]]; then
  echo "[railway_start_gunicorn] Bootstrapping PHI redactor vendor bundle from S3..."
  python ops/tools/bootstrap_phi_redactor_vendor_bundle.py
fi

# Optional: bootstrap registry model bundle from S3 before app starts.
# The FastAPI lifespan validator requires a populated runtime bundle when MODEL_BACKEND=onnx.
if [[ -n "${MODEL_BUNDLE_S3_URI_ONNX:-${MODEL_BUNDLE_S3_URI_PYTORCH:-${MODEL_BUNDLE_S3_URI:-}}}" ]]; then
  echo "[railway_start_gunicorn] Bootstrapping registry model bundle from S3..."
  python - <<'PY'
from app.registry.model_bootstrap import ensure_registry_model_bundle
ensure_registry_model_bundle()
PY
fi

# NOTE: `uvicorn.workers.UvicornWorker` is the traditional integration; check uvicorn docs for the
# recommended worker package/version for your deployment.
exec gunicorn "app.api.fastapi_app:app" \
  --bind "0.0.0.0:${PORT}" \
  --workers "${WORKERS}" \
  --worker-class "uvicorn.workers.UvicornWorker" \
  --preload \
  --timeout "${TIMEOUT}"

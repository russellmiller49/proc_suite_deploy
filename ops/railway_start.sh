#!/usr/bin/env bash
# Railway startup script for Procedure Suite API
#
# This script:
# 1. Starts the FastAPI application quickly (liveness via /health)
# 2. Lets the app perform background warmup (readiness via /ready)
#
# Usage:
#   ops/railway_start.sh
#
# Environment variables:
#   PORT - Port to listen on (default: 8000)
#   PROCSUITE_SPACY_MODEL - spaCy model to use (default: en_core_sci_sm)
#   WORKERS - Number of uvicorn workers (default: 1)
#   LIMIT_CONCURRENCY - uvicorn concurrency cap (default: 50)
#
# Railway Configuration:
#   Set "Start Command" to: ops/railway_start.sh

set -euo pipefail

# Ensure Python can find the local packages (modules, proc_*, etc.)
# Railway runs from /app with a virtual environment, but PYTHONPATH may not include /app
export PYTHONPATH="${PYTHONPATH:-}:${PWD}"
echo "[railway_start] PYTHONPATH=${PYTHONPATH}"

echo "[railway_start] =============================================="
echo "[railway_start] Starting Procedure Suite API"
echo "[railway_start] =============================================="
echo "[railway_start] PORT=${PORT:-8000}"
echo "[railway_start] PROCSUITE_SPACY_MODEL=${PROCSUITE_SPACY_MODEL:-en_core_sci_sm}"
echo "[railway_start] WORKERS=${WORKERS:-1}"
echo "[railway_start] LIMIT_CONCURRENCY=${LIMIT_CONCURRENCY:-50}"
export MODEL_BACKEND="${MODEL_BACKEND:-onnx}"
echo "[railway_start] MODEL_BACKEND=${MODEL_BACKEND}"

# Limit BLAS/OpenMP thread oversubscription (important for sklearn/ONNX on small Railway machines)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"
echo "[railway_start] OMP_NUM_THREADS=${OMP_NUM_THREADS} MKL_NUM_THREADS=${MKL_NUM_THREADS} OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS}"
echo "[railway_start] =============================================="

echo "[railway_start] =============================================="
echo "[railway_start] Starting FastAPI (uvicorn)..."
echo "[railway_start] =============================================="

# Optional: run Alembic migrations on start (recommended for single-instance Railway deploys).
if [[ "${PROCSUITE_RUN_MIGRATIONS_ON_START:-}" =~ ^(1|true|yes)$ ]]; then
  echo "[railway_start] Running migrations (alembic upgrade head)..."
  alembic upgrade head
fi

# Optional: bootstrap PHI redactor vendor bundle from S3 before app starts.
if [[ -n "${PHI_REDACTOR_VENDOR_BUNDLE_S3_URI:-${PHI_REDACTOR_VENDOR_BUNDLE_S3_URI_ONNX:-}}" ]]; then
  echo "[railway_start] Bootstrapping PHI redactor vendor bundle from S3..."
  python ops/tools/bootstrap_phi_redactor_vendor_bundle.py
fi

# Optional: bootstrap registry model bundle from S3 before app starts.
# The FastAPI lifespan validator requires a populated runtime bundle when MODEL_BACKEND=onnx.
if [[ -n "${MODEL_BUNDLE_S3_URI_ONNX:-${MODEL_BUNDLE_S3_URI_PYTORCH:-${MODEL_BUNDLE_S3_URI:-}}}" ]]; then
  echo "[railway_start] Bootstrapping registry model bundle from S3..."
  python - <<'PY'
from app.registry.model_bootstrap import ensure_registry_model_bundle
ensure_registry_model_bundle()
PY
fi

# Step 2: Start uvicorn
# Using exec to replace the shell process with uvicorn
# This ensures proper signal handling for graceful shutdown
# Use 'python -m uvicorn' to ensure we use the conda environment's Python
exec python -m uvicorn app.api.fastapi_app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --workers "${WORKERS:-1}" \
    --limit-concurrency "${LIMIT_CONCURRENCY:-50}" \
    --log-level info

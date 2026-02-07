#!/usr/bin/env bash
# Local development server for Procedure Suite API
#
# This script mirrors railway_start.sh but with hot-reload enabled.
# Uses the same optimization settings for consistent behavior.
#
# Usage:
#   ./ops/devserver.sh
#
# Environment variables (all optional, sensible defaults provided):
#   PORT - Port to listen on (default: 8000)
#   SKIP_WARMUP - Skip model warmup for faster startup (default: false)
#   ENABLE_UMLS_LINKER - Load UMLS linker (default: true, set false to save RAM)

set -euo pipefail

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Fast mode toggle (keeps extraction deterministic and avoids self-correction LLM calls)
if [[ "${PROCSUITE_FAST_MODE:-0}" == "1" ]]; then
  export REGISTRY_SELF_CORRECT_ENABLED="0"
fi

# Knowledge base
# - Override via `PSUITE_KNOWLEDGE_FILE` (preferred) or `CODER_KB_PATH` (legacy).
# - Defaults are resolved in `config/settings.py:KnowledgeSettings`.

# Enable LLM Advisor for dev server to verify reliability fixes
export CODER_USE_LLM_ADVISOR="${CODER_USE_LLM_ADVISOR:-true}"

# Model backend:
# - Prefer explicit shell env MODEL_BACKEND if set
# - Else prefer .env MODEL_BACKEND (sourced above)
# - Else default to pytorch for local dev (avoids requiring an ONNX runtime bundle)
export MODEL_BACKEND="${MODEL_BACKEND:-pytorch}"

# Limit BLAS/OpenMP thread oversubscription (matches Railway settings)
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export VECLIB_MAXIMUM_THREADS="${VECLIB_MAXIMUM_THREADS:-1}"

# Concurrency settings
export LIMIT_CONCURRENCY="${LIMIT_CONCURRENCY:-50}"
export LLM_CONCURRENCY="${LLM_CONCURRENCY:-2}"
export CPU_WORKERS="${CPU_WORKERS:-1}"
export ENABLE_UMLS_LINKER="${ENABLE_UMLS_LINKER:-true}"

echo "[devserver] =============================================="
echo "[devserver] Starting Procedure Suite API (dev mode)"
echo "[devserver] =============================================="
echo "[devserver] PORT=${PORT:-8000}"
echo "[devserver] MODEL_BACKEND=${MODEL_BACKEND}"
echo "[devserver] PSUITE_KNOWLEDGE_FILE=${PSUITE_KNOWLEDGE_FILE-<unset>}"
echo "[devserver] ENABLE_UMLS_LINKER=${ENABLE_UMLS_LINKER}"
echo "[devserver] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[devserver] =============================================="

# Use 'python -m uvicorn' to ensure we use the conda environment's Python
# --reload enables hot-reload for development
exec python -m uvicorn app.api.fastapi_app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --limit-concurrency "${LIMIT_CONCURRENCY}" \
    --reload \
    --log-level info

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
#   UMLS_ENABLE_LINKER (alias: ENABLE_UMLS_LINKER) - Enable UMLS integration (default: true)
#   UMLS_LINKER_BACKEND - distilled (default) | scispacy
#   UMLS_IP_UMLS_MAP_LOCAL_PATH (alias: IP_UMLS_MAP_PATH) - Explicit local distilled map override
#   UMLS_IP_UMLS_MAP_S3_URI - S3 URI for distilled map (downloaded to cache)
#   UMLS_IP_UMLS_MAP_CACHE_PATH - Cache path for S3 map (default: /tmp/procsuite/ip_umls_map.json)
#   UMLS_FORCE_REFRESH - Redownload S3 map on boot (default: false)

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

# UMLS settings (default: distilled deterministic linker)
export UMLS_ENABLE_LINKER="${UMLS_ENABLE_LINKER:-${ENABLE_UMLS_LINKER:-true}}"
export ENABLE_UMLS_LINKER="${ENABLE_UMLS_LINKER:-${UMLS_ENABLE_LINKER}}"
export UMLS_LINKER_BACKEND="${UMLS_LINKER_BACKEND:-distilled}"
if [[ -z "${UMLS_IP_UMLS_MAP_LOCAL_PATH-}" && -n "${IP_UMLS_MAP_PATH-}" ]]; then
  export UMLS_IP_UMLS_MAP_LOCAL_PATH="${IP_UMLS_MAP_PATH}"
fi
if [[ -z "${IP_UMLS_MAP_PATH-}" && -n "${UMLS_IP_UMLS_MAP_LOCAL_PATH-}" ]]; then
  export IP_UMLS_MAP_PATH="${UMLS_IP_UMLS_MAP_LOCAL_PATH}"
fi
export UMLS_IP_UMLS_MAP_CACHE_PATH="${UMLS_IP_UMLS_MAP_CACHE_PATH:-/tmp/procsuite/ip_umls_map.json}"
export UMLS_FORCE_REFRESH="${UMLS_FORCE_REFRESH:-false}"

echo "[devserver] =============================================="
echo "[devserver] Starting Procedure Suite API (dev mode)"
echo "[devserver] =============================================="
echo "[devserver] PORT=${PORT:-8000}"
echo "[devserver] MODEL_BACKEND=${MODEL_BACKEND}"
echo "[devserver] PSUITE_KNOWLEDGE_FILE=${PSUITE_KNOWLEDGE_FILE-<unset>}"
echo "[devserver] UMLS_ENABLE_LINKER=${UMLS_ENABLE_LINKER}"
echo "[devserver] UMLS_LINKER_BACKEND=${UMLS_LINKER_BACKEND}"
echo "[devserver] UMLS_IP_UMLS_MAP_LOCAL_PATH=${UMLS_IP_UMLS_MAP_LOCAL_PATH-<unset>}"
echo "[devserver] UMLS_IP_UMLS_MAP_S3_URI=${UMLS_IP_UMLS_MAP_S3_URI-<unset>}"
echo "[devserver] UMLS_IP_UMLS_MAP_CACHE_PATH=${UMLS_IP_UMLS_MAP_CACHE_PATH}"
echo "[devserver] UMLS_FORCE_REFRESH=${UMLS_FORCE_REFRESH}"
echo "[devserver] ENABLE_UMLS_LINKER=${ENABLE_UMLS_LINKER} (legacy alias)"
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

#!/usr/bin/env bash
# Local development server for Procedure Suite API
#
# This script mirrors railway_start.sh but with hot-reload enabled.
# Uses the same optimization settings for consistent behavior.
#
# Usage:
#   ./ops/devserver.sh
#   ./ops/devserver.sh --ui=atlas
#   ./ops/devserver.sh --ui=classic
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

UI_VARIANT="${PROCSUITE_UI_VARIANT:-atlas}"
for arg in "$@"; do
  case "$arg" in
    --ui=classic)
      UI_VARIANT="classic"
      ;;
    --ui=atlas)
      UI_VARIANT="atlas"
      ;;
    --ui=*)
      echo "[devserver] ERROR: unsupported UI variant flag '$arg' (use --ui=atlas or --ui=classic)" >&2
      exit 1
      ;;
    *)
      echo "[devserver] ERROR: unsupported argument '$arg'" >&2
      exit 1
      ;;
  esac
done
export PROCSUITE_UI_VARIANT="${UI_VARIANT}"

if [[ -f ".env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi
export PROCSUITE_UI_VARIANT="${UI_VARIANT}"

# Keep runtime writes out of tracked repo DB files during local dev.
# If DATABASE_URL is provided, respect it. Otherwise default PHI and registry
# persistence to an untracked sqlite file under /tmp.
DEV_SQLITE_DB_FILE="${PROCSUITE_DEV_SQLITE_DB_FILE:-/tmp/procsuite/procsuite_dev.db}"
PHI_DB_DEFAULTED=0
REGISTRY_STORE_DB_DEFAULTED=0
if [[ -z "${DATABASE_URL-}" ]]; then
  mkdir -p "$(dirname "${DEV_SQLITE_DB_FILE}")"
  DEV_SQLITE_DB_URL="sqlite:////${DEV_SQLITE_DB_FILE#/}"
  if [[ -z "${PHI_DATABASE_URL-}" ]]; then
    export PHI_DATABASE_URL="${DEV_SQLITE_DB_URL}"
    PHI_DB_DEFAULTED=1
  fi
  if [[ -z "${REGISTRY_STORE_DATABASE_URL-}" ]]; then
    export REGISTRY_STORE_DATABASE_URL="${DEV_SQLITE_DB_URL}"
    REGISTRY_STORE_DB_DEFAULTED=1
  fi
fi

# Local UI vault flow uses X-User-Id headers.
# Default this on for devserver even if .env enables production gating.
export VAULT_AUTH_ALLOW_X_USER_ID="${VAULT_AUTH_ALLOW_X_USER_ID:-true}"
# Local Vault longitudinal workflows require persisted registry runs.
export REGISTRY_RUNS_PERSIST_ENABLED="${REGISTRY_RUNS_PERSIST_ENABLED:-1}"

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
echo "[devserver] VAULT_AUTH_ALLOW_X_USER_ID=${VAULT_AUTH_ALLOW_X_USER_ID}"
echo "[devserver] REGISTRY_RUNS_PERSIST_ENABLED=${REGISTRY_RUNS_PERSIST_ENABLED}"
echo "[devserver] PROCSUITE_UI_VARIANT=${PROCSUITE_UI_VARIANT}"
if [[ "${PHI_DB_DEFAULTED}" == "1" ]]; then
  echo "[devserver] PHI_DATABASE_URL=${PHI_DATABASE_URL} (local dev default)"
elif [[ -n "${PHI_DATABASE_URL-}" ]]; then
  if [[ "${PHI_DATABASE_URL}" == sqlite* ]]; then
    echo "[devserver] PHI_DATABASE_URL=${PHI_DATABASE_URL}"
  else
    echo "[devserver] PHI_DATABASE_URL=<set non-sqlite>"
  fi
else
  echo "[devserver] PHI_DATABASE_URL=<unset>"
fi
if [[ "${REGISTRY_STORE_DB_DEFAULTED}" == "1" ]]; then
echo "[devserver] REGISTRY_STORE_DATABASE_URL=${REGISTRY_STORE_DATABASE_URL} (local dev default)"
elif [[ -n "${REGISTRY_STORE_DATABASE_URL-}" ]]; then
  if [[ "${REGISTRY_STORE_DATABASE_URL}" == sqlite* ]]; then
    echo "[devserver] REGISTRY_STORE_DATABASE_URL=${REGISTRY_STORE_DATABASE_URL}"
  else
    echo "[devserver] REGISTRY_STORE_DATABASE_URL=<set non-sqlite>"
  fi
else
  echo "[devserver] REGISTRY_STORE_DATABASE_URL=<unset>"
fi
echo "[devserver] OMP_NUM_THREADS=${OMP_NUM_THREADS}"
echo "[devserver] Open UI at http://localhost:${PORT:-8000}/ui/"
echo "[devserver] Reporter Builder: http://localhost:${PORT:-8000}/ui/reporter_builder.html"
echo "[devserver] =============================================="

# Fast fail with a clear message when new API dependencies are missing from the
# interpreter that will actually launch uvicorn.
if ! python - <<'PY' >/dev/null 2>&1
import multipart
PY
then
  echo "[devserver] ERROR: python-multipart is not installed in the active Python environment." >&2
  echo "[devserver] python=$(python -c 'import sys; print(sys.executable)')" >&2
  echo "[devserver] Install updated runtime deps in this environment, for example:" >&2
  echo "[devserver]   python -m pip install -r requirements.txt" >&2
  exit 1
fi

# Use 'python -m uvicorn' to ensure we use the conda environment's Python
# --reload enables hot-reload for development
exec python -m uvicorn app.api.fastapi_app:app \
    --host 0.0.0.0 \
    --port "${PORT:-8000}" \
    --limit-concurrency "${LIMIT_CONCURRENCY}" \
    --reload \
    --log-level info

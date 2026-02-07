#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ALLOWLIST_FILE="${DEPLOY_MIRROR_ALLOWLIST:-${REPO_ROOT}/ops/deploy/mirror_paths.txt}"
DEST_DIR="${1:-${REPO_ROOT}/.tmp/deploy_mirror}"
INCLUDE_VENDOR="${DEPLOY_MIRROR_INCLUDE_VENDOR:-0}"

if [[ ! -f "${ALLOWLIST_FILE}" ]]; then
  echo "[build_deploy_mirror] ERROR: allowlist not found: ${ALLOWLIST_FILE}" >&2
  exit 1
fi

echo "[build_deploy_mirror] allowlist=${ALLOWLIST_FILE}"
echo "[build_deploy_mirror] dest=${DEST_DIR}"
echo "[build_deploy_mirror] include_phi_vendor=${INCLUDE_VENDOR}"

rm -rf "${DEST_DIR}"
mkdir -p "${DEST_DIR}"

missing_count=0

copy_allowlisted_path() {
  local rel_path="$1"
  local src_path="${REPO_ROOT}/${rel_path}"
  local dst_path="${DEST_DIR}/${rel_path}"

  if [[ ! -e "${src_path}" ]]; then
    echo "[build_deploy_mirror] ERROR: missing allowlisted path: ${rel_path}" >&2
    missing_count=$((missing_count + 1))
    return
  fi

  mkdir -p "$(dirname "${dst_path}")"

  if [[ -d "${src_path}" ]]; then
    local -a rsync_args=(
      -a
      --exclude ".DS_Store"
      --exclude "__pycache__/"
      --exclude "*.pyc"
      --exclude ".pytest_cache/"
      --exclude ".mypy_cache/"
      --exclude ".ruff_cache/"
    )
    if [[ "${rel_path}" == "ui/static" && "${INCLUDE_VENDOR}" != "1" ]]; then
      rsync_args+=(--exclude "phi_redactor/vendor/")
    fi
    rsync "${rsync_args[@]}" "${src_path}/" "${dst_path}/"
  else
    cp "${src_path}" "${dst_path}"
  fi
}

while IFS= read -r raw_line || [[ -n "${raw_line}" ]]; do
  line="$(printf '%s' "${raw_line}" | sed -E 's/[[:space:]]*#.*$//')"
  line="$(printf '%s' "${line}" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//')"
  [[ -z "${line}" ]] && continue
  copy_allowlisted_path "${line}"
done < "${ALLOWLIST_FILE}"

if [[ "${missing_count}" -gt 0 ]]; then
  echo "[build_deploy_mirror] ERROR: ${missing_count} allowlisted paths were missing." >&2
  exit 1
fi

find "${DEST_DIR}" -type d -name "__pycache__" -prune -exec rm -rf {} +
find "${DEST_DIR}" -type d \( -name ".pytest_cache" -o -name ".mypy_cache" -o -name ".ruff_cache" \) -prune -exec rm -rf {} +
find "${DEST_DIR}" -type f \( -name "*.pyc" -o -name ".DS_Store" \) -delete
find "${DEST_DIR}" -type d -empty -delete

echo "[build_deploy_mirror] Built deploy mirror payload:"
du -sh "${DEST_DIR}"

#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
BACKEND="${2:-pytorch}"
OUT_DIR="${3:-dist/registry_bundle}"

if [[ -z "${VERSION}" ]]; then
  echo "Usage: $0 <version> [pytorch|onnx] [out_dir]" >&2
  exit 2
fi

if [[ "${BACKEND}" != "pytorch" && "${BACKEND}" != "onnx" ]]; then
  echo "BACKEND must be 'pytorch' or 'onnx' (got: ${BACKEND})" >&2
  exit 2
fi

TARBALL="${OUT_DIR}/bundle.tar.gz"
MANIFEST="${OUT_DIR}/manifest.json"

if [[ ! -f "${TARBALL}" ]]; then
  echo "Missing bundle tarball at ${TARBALL}" >&2
  exit 2
fi

if [[ ! -f "${MANIFEST}" ]]; then
  echo "Missing manifest at ${MANIFEST}" >&2
  exit 2
fi

DEST="s3://procedure-suite-models/deploy/registry/${VERSION}/${BACKEND}/"

echo "Uploading to: ${DEST}"
aws s3 cp "${TARBALL}" "${DEST}bundle.tar.gz"
aws s3 cp "${MANIFEST}" "${DEST}manifest.json"
echo "Done."


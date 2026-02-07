#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${MODEL_BUNDLE_S3_URI_PYTORCH:-}" ]]; then
  echo "Missing MODEL_BUNDLE_S3_URI_PYTORCH (example: s3://.../pytorch/bundle.tar.gz OR s3://.../run_prefix/)" >&2
  exit 2
fi

DEST_DIR="${REGISTRY_RUNTIME_DIR:-data/models/registry_runtime}"
URI="${MODEL_BUNDLE_S3_URI_PYTORCH}"
export REGISTRY_RUNTIME_DIR="${DEST_DIR}"

echo "[dev_pull_model] Downloading: ${URI}"
echo "[dev_pull_model] Extracting to: ${DEST_DIR}"

TMPDIR="$(mktemp -d)"
trap 'rm -rf "${TMPDIR}"' EXIT

if [[ "${URI}" == *.tar.gz ]]; then
  TARBALL="${TMPDIR}/bundle.tar.gz"
  aws s3 cp "${URI}" "${TARBALL}" --only-show-errors

  rm -rf "${DEST_DIR}"
  mkdir -p "${DEST_DIR}"
  tar -xzf "${TARBALL}" -C "${DEST_DIR}"

  echo "[dev_pull_model] Done."
  exit 0
fi

PREFIX="${URI%/}/"
BUNDLE_PREFIX="${PREFIX}"

# Prefer the training-run layout where torch artifacts live under model/.
if aws s3 ls "${PREFIX}model/" >/dev/null 2>&1; then
  BUNDLE_PREFIX="${PREFIX}model/"
fi

bundle_root="${TMPDIR}/bundle"
mkdir -p "${bundle_root}/tokenizer"

try_copy() {
  local dest="$1"
  shift
  local src
  for src in "$@"; do
    if aws s3 cp "${src}" "${dest}" --only-show-errors >/dev/null 2>&1; then
      echo "${src}"
      return 0
    fi
  done
  return 1
}

config_src="$(try_copy "${bundle_root}/config.json" "${BUNDLE_PREFIX}config.json" "${PREFIX}config.json" || true)"
if [[ -z "${config_src}" ]]; then
  echo "[dev_pull_model] ERROR: missing config.json under ${URI}" >&2
  exit 1
fi

weights_src="$(try_copy "${bundle_root}/model.safetensors" "${BUNDLE_PREFIX}model.safetensors" || true)"
if [[ -z "${weights_src}" ]]; then
  weights_src="$(try_copy "${bundle_root}/pytorch_model.bin" "${BUNDLE_PREFIX}pytorch_model.bin" || true)"
fi
if [[ -z "${weights_src}" ]]; then
  echo "[dev_pull_model] ERROR: missing model weights under ${URI} (expected model.safetensors or pytorch_model.bin)" >&2
  exit 1
fi

classifier_src="$(try_copy "${bundle_root}/classifier.pt" "${BUNDLE_PREFIX}classifier.pt" || true)"
if [[ -z "${classifier_src}" ]]; then
  echo "[dev_pull_model] ERROR: missing classifier.pt under ${URI}" >&2
  exit 1
fi

thresholds_src="$(try_copy "${bundle_root}/thresholds.json" "${BUNDLE_PREFIX}thresholds.json" "${PREFIX}thresholds.json" || true)"
if [[ -z "${thresholds_src}" ]]; then
  echo "[dev_pull_model] ERROR: missing thresholds.json under ${URI}" >&2
  exit 1
fi

label_src="$(try_copy "${bundle_root}/label_order.json" "${BUNDLE_PREFIX}label_order.json" "${PREFIX}label_order.json" "${BUNDLE_PREFIX}registry_label_fields.json" "${PREFIX}registry_label_fields.json" || true)"
if [[ -z "${label_src}" ]]; then
  echo "[dev_pull_model] ERROR: missing label_order.json/registry_label_fields.json under ${URI}" >&2
  exit 1
fi

# Ensure both filenames exist for downstream code.
cp "${bundle_root}/label_order.json" "${bundle_root}/registry_label_fields.json"

tokenizer_src=""
for cand in "${BUNDLE_PREFIX}tokenizer/" "${PREFIX}tokenizer/"; do
  if aws s3 ls "${cand}" >/dev/null 2>&1; then
    tokenizer_src="${cand}"
    break
  fi
done
if [[ -z "${tokenizer_src}" ]]; then
  echo "[dev_pull_model] ERROR: missing tokenizer/ directory under ${URI}" >&2
  exit 1
fi
aws s3 sync "${tokenizer_src}" "${bundle_root}/tokenizer" --only-show-errors

model_version="${PREFIX%/}"
model_version="${model_version##*/}"
cat >"${bundle_root}/manifest.json" <<EOF
{
  "model_backend": "pytorch",
  "model_version": "${model_version}",
  "source_uri": "${PREFIX}",
  "configured_source_uri": "${URI}",
  "source_type": "s3_prefix"
}
EOF

rm -rf "${DEST_DIR}"
mkdir -p "$(dirname "${DEST_DIR}")"
mv "${bundle_root}" "${DEST_DIR}"

echo "[dev_pull_model] Verifying torch bundle..."
python - <<'PY'
import os
from app.registry.inference_pytorch import TorchRegistryPredictor

bundle_dir = os.environ.get("REGISTRY_RUNTIME_DIR") or "data/models/registry_runtime"
pred = TorchRegistryPredictor(bundle_dir=bundle_dir)
print("[dev_pull_model] TorchRegistryPredictor.available =", pred.available)
print("[dev_pull_model] labels =", len(pred.labels))
PY

echo "[dev_pull_model] Done."

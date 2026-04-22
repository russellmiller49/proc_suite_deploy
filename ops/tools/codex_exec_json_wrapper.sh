#!/usr/bin/env bash
set -euo pipefail

# Wrapper for deterministic workflow integration:
# - runs Codex in non-interactive mode (`codex exec`)
# - writes only the final assistant message to stdout
#   so the Python runner can parse strict JSON output

CODEX_BIN="${CODEX_BIN:-/home/rjm/.npm-global/bin/codex}"
if [[ ! -x "$CODEX_BIN" ]]; then
  CODEX_BIN="$(command -v codex || true)"
fi

if [[ -z "$CODEX_BIN" ]]; then
  echo "codex binary not found. Set CODEX_BIN or install codex CLI." >&2
  exit 1
fi

LAST_MSG_FILE="$(mktemp)"
cleanup() {
  rm -f "$LAST_MSG_FILE"
}
trap cleanup EXIT

"$CODEX_BIN" exec --output-last-message "$LAST_MSG_FILE" "$@"
cat "$LAST_MSG_FILE"

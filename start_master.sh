#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="$ROOT_DIR/.venv_master_dfs"
CONFIG_PATH="${CONFIG_PATH:-$ROOT_DIR/config_extended.json}"
MASTER_HOST="${MASTER_HOST:-0.0.0.0}"
MASTER_PORT="${MASTER_PORT:-18080}"
MASTER_URL="http://127.0.0.1:${MASTER_PORT}"

"$PYTHON_BIN" - <<'PY'
import os
import sys

allow_unsupported = os.environ.get("ALLOW_UNSUPPORTED_PYTHON") == "1"
if sys.version_info < (3, 14) and not allow_unsupported:
    raise SystemExit("Python 3.14+ is required for start_master.sh. Set ALLOW_UNSUPPORTED_PYTHON=1 only for local development smoke tests.")
PY

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip
python -m pip install -r "$ROOT_DIR/master/requirements_extended.txt"

if command -v open >/dev/null 2>&1; then
  (sleep 3 && open "$MASTER_URL") &
elif command -v xdg-open >/dev/null 2>&1; then
  (sleep 3 && xdg-open "$MASTER_URL") &
fi

exec python -m master.master_dfs --config "$CONFIG_PATH" --host "$MASTER_HOST" --port "$MASTER_PORT" --log-level INFO --auto-start

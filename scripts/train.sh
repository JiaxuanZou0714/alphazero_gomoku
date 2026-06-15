#!/usr/bin/env bash
# Linux/WSL launcher for a training preset. Replaces the old per-preset .cmd files.
#
#   scripts/train.sh <preset> [extra --flags ...]
#
# Logs to outputs/logs/<preset>_train.{out,err}.log. Activates the project venv
# (~/.venvs/azg) if present. Runs from the repo's parent dir so the presets'
# "alphazero_gomoku/outputs/..." relative paths resolve.
set -euo pipefail

PRESET="${1:?usage: scripts/train.sh <preset> [extra args]}"
shift || true

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"
LOG_DIR="$REPO_DIR/outputs/logs"
mkdir -p "$LOG_DIR"

if [ -f "$HOME/.venvs/azg/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$HOME/.venvs/azg/bin/activate"
fi

cd "$PARENT_DIR"
exec python -m alphazero_gomoku.train --preset "$PRESET" "$@" \
  > "$LOG_DIR/${PRESET}_train.out.log" 2> "$LOG_DIR/${PRESET}_train.err.log"

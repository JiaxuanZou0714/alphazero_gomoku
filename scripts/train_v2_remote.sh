#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PARENT_DIR="$(dirname "$REPO_DIR")"
LOG_DIR="$REPO_DIR/outputs/logs"
mkdir -p "$LOG_DIR"

resume="$REPO_DIR/outputs/checkpoints/a100-4-prod-v3/gomoku10_best.pt"
latest_v2="$(find "$REPO_DIR/outputs/checkpoints/v2" -maxdepth 1 -name 'gomoku10_iter_*.pt' -print 2>/dev/null | sort | tail -n 1 || true)"
if [[ -n "$latest_v2" ]]; then
  resume="$latest_v2"
fi

cd "$PARENT_DIR"
exec python -m alphazero_gomoku.train \
  --preset v2 \
  --resume "$resume" \
  --self-play-workers "${SELF_PLAY_WORKERS:-32}" \
  "$@"

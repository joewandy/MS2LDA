#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SOURCE_RUN="${HYBRID_SIMPLIFICATION_SOURCE_RUN:-/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/indicative-msnlib-k1000-seed42-peak-pooling-correction}"
RUN_DIR="${HYBRID_SIMPLIFICATION_RUN_DIR:-/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/hybrid-lda-simplification-seed42}"
SESSION="${HYBRID_SIMPLIFICATION_SCREEN_SESSION:-hybridlda-simplification}"
MODULE="benchmarks.msnlib_simplification"

usage() {
  echo "Usage: $0 {start|resume|status|verify|verify-archive}"
}

module_command() {
  conda run --no-capture-output -n ms2lda-hybrid \
    python -m "$MODULE" "$@" --run "$RUN_DIR"
}

screen_active() {
  local sessions
  sessions="$(/usr/bin/screen -ls 2>/dev/null || true)"
  [[ "$sessions" == *".${SESSION}"* ]]
}

start_runner() {
  if [[ ! -f "$RUN_DIR/simplification.lock.json" ]]; then
    mkdir -p "$RUN_DIR"
    conda run --no-capture-output -n ms2lda-hybrid \
      python -m "$MODULE" freeze \
      --run "$RUN_DIR" \
      --source-run "$SOURCE_RUN" \
      --repo-root "$REPO_ROOT"
  fi
  if screen_active; then
    echo "Screen session $SESSION is already active."
    module_command status
    exit 1
  fi
  module_command preflight
  mkdir -p "$RUN_DIR/logs"
  /usr/bin/screen -DmS "$SESSION" \
    env SIMPLIFICATION_REPO="$REPO_ROOT" SIMPLIFICATION_RUN="$RUN_DIR" \
    /bin/zsh -lc \
    'cd "$SIMPLIFICATION_REPO" && exec /usr/bin/caffeinate -i conda run --no-capture-output -n ms2lda-hybrid python -m benchmarks.msnlib_simplification run --run "$SIMPLIFICATION_RUN" >> "$SIMPLIFICATION_RUN/overnight.log" 2>&1' \
    </dev/null >/dev/null 2>&1 &

  for _ in {1..30}; do
    if [[ -f "$RUN_DIR/heartbeat.json" ]] && screen_active; then
      echo "Overnight runner launched in screen session $SESSION."
      echo "Keep the Mac plugged in with its lid open."
      module_command status
      return
    fi
    sleep 1
  done
  echo "Runner did not publish a healthy heartbeat within 30 seconds."
  [[ -f "$RUN_DIR/overnight.log" ]] && tail -n 80 "$RUN_DIR/overnight.log"
  exit 1
}

case "${1:-}" in
  start|resume)
    start_runner
    ;;
  status)
    module_command status
    ;;
  verify)
    module_command verify
    ;;
  verify-archive)
    module_command verify-archive
    ;;
  *)
    usage
    exit 2
    ;;
esac

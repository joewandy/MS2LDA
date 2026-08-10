#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/hybrid-lda-simplification-seed42-v8"
REPO_ROOT="/Users/joewandy/Work/git/MS2LDA-hybrid-simplification"
CHEMICAL_LOG="$RUN_DIR/logs/chemical_scores.log"
RECOVERY_LOG="$RUN_DIR/recovery.log"
HYBRID_PYTHON="/Users/joewandy/anaconda3/envs/ms2lda-hybrid/bin/python"
RECOVERY_DRIVER="$RUN_DIR/recover_chemical_scores.py"

cd "$REPO_ROOT"
printf '\n[%s] recovery attempt 5 in frozen ms2lda-hybrid environment\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$CHEMICAL_LOG"
started_epoch="$(date +%s)"
started_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
  "$HYBRID_PYTHON" "$RECOVERY_DRIVER" "$RUN_DIR" \
  >> "$CHEMICAL_LOG" 2>&1

completed_epoch="$(date +%s)"
completed_utc="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
elapsed_seconds="$((completed_epoch - started_epoch))"

/usr/bin/python3 - "$RUN_DIR" "$HYBRID_PYTHON" "$RECOVERY_DRIVER" \
  "$started_utc" "$completed_utc" "$elapsed_seconds" <<'PY'
import json
import os
import sys
from pathlib import Path

run_dir = Path(sys.argv[1])
hybrid_python = sys.argv[2]
recovery_driver = sys.argv[3]
started_utc = sys.argv[4]
completed_utc = sys.argv[5]
elapsed_seconds = float(sys.argv[6])
state_path = run_dir / "run_state.json"
state = json.loads(state_path.read_text(encoding="utf-8"))
state["tasks"]["chemical_scores"] = {
    "status": "completed",
    "started_utc": started_utc,
    "completed_utc": completed_utc,
    "command": [
        hybrid_python,
        recovery_driver,
        str(run_dir),
    ],
    "environment": "ms2lda-hybrid",
    "recovery_driver": str(Path(recovery_driver)),
    "recovery_reason": (
        "verify completed annotations without re-importing their build-only "
        "FAISS and gensim dependencies"
    ),
    "aborted_recovery_attempts": [2, 3, 4],
    "log": "logs/chemical_scores.log",
    "attempt": 5,
    "elapsed_seconds": elapsed_seconds,
    "returncode": 0,
}
temporary = state_path.with_name(f".{state_path.name}.recovery.tmp")
with temporary.open("w", encoding="utf-8") as handle:
    json.dump(state, handle, indent=2, sort_keys=True)
    handle.write("\n")
    handle.flush()
    os.fsync(handle.fileno())
temporary.replace(state_path)

event_path = run_dir / "events.jsonl"
event = {
    "created_utc": completed_utc,
    "event": "completed",
    "task": "chemical_scores",
    "returncode": 0,
    "attempt": 5,
    "environment": "ms2lda-hybrid",
    "recovery_reason": "verified existing annotations without build-only imports",
}
with event_path.open("a", encoding="utf-8") as handle:
    handle.write(json.dumps(event, sort_keys=True, separators=(",", ":")) + "\n")
    handle.flush()
    os.fsync(handle.fileno())
PY

printf '[%s] chemical recovery completed; resuming runner\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$RECOVERY_LOG"
exec "$HYBRID_PYTHON" -m benchmarks.msnlib_simplification run \
  --run "$RUN_DIR" >> "$RUN_DIR/overnight.log" 2>&1

#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

action="${1:-status}"
run_root="${NEURAL_MS2LDA_RUN:?set NEURAL_MS2LDA_RUN to the run directory}"
data_root="${NEURAL_MS2LDA_DATA:?set NEURAL_MS2LDA_DATA to the acquired MSnLib root}"
reference_root="${NEURAL_MS2LDA_TOMOTOPY_REFERENCE:?set NEURAL_MS2LDA_TOMOTOPY_REFERENCE to the frozen Tomotopy run}"
environment_name="${NEURAL_MS2LDA_ENV:-ms2lda-neural}"
pid_file="$run_root/runner.pid"
log_file="$run_root/logs/runner.log"

export OMP_NUM_THREADS=6
export MKL_NUM_THREADS=6
export OPENBLAS_NUM_THREADS=6
export VECLIB_MAXIMUM_THREADS=6
export NUMEXPR_NUM_THREADS=6

case "$action" in
  start)
    mkdir -p "$run_root/logs"
    if [[ -f "$pid_file" ]]; then
      existing_pid="$(<"$pid_file")"
      if kill -0 "$existing_pid" 2>/dev/null; then
        echo "runner process already exists: ${existing_pid}" >&2
        exit 1
      fi
    fi
    command=(
      conda run --no-capture-output -n "$environment_name"
      python -m benchmarks.neural_assignment_ms2lda run
      --data-root "$data_root" --run "$run_root"
      --tomotopy-reference-run "$reference_root"
    )
    nohup "${command[@]}" >>"$log_file" 2>&1 </dev/null &
    runner_pid=$!
    printf '%s\n' "$runner_pid" >"$pid_file"
    echo "started process ${runner_pid}; inspect with: $0 status"
    ;;
  resume)
    "$0" start
    ;;
  status)
    conda run --no-capture-output -n "$environment_name" \
      python -m benchmarks.neural_assignment_ms2lda status --run "$run_root"
    ;;
  verify)
    conda run --no-capture-output -n "$environment_name" \
      python -m benchmarks.neural_assignment_ms2lda verify \
      --data-root "$data_root" --run "$run_root"
    ;;
  attach)
    tail -f "$log_file"
    ;;
  *)
    echo "usage: $0 {start|resume|status|verify|attach}" >&2
    exit 2
    ;;
esac

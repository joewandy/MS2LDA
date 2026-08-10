#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/joewandy/Work/git/MS2LDA"
DEFAULT_DATA_ROOT="/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs"
SOURCE_RUN="${FULLY_NEURAL_SOURCE_RUN:-${DEFAULT_DATA_ROOT}/hybrid-lda-simplification-seed42-v8}"
REFERENCE_RUN="${FULLY_NEURAL_REFERENCE_RUN:-${DEFAULT_DATA_ROOT}/indicative-msnlib-k1000-seed42-peak-pooling-correction}"
RUN_DIR="${FULLY_NEURAL_RUN_DIR:-${DEFAULT_DATA_ROOT}/fully-neural-ms2lda-seed42-v1}"
SESSION_NAME="fully-neural-ms2lda-seed42-v1"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/runner.log"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

foreground() {
  mkdir -p "${LOG_DIR}"
  cd "${REPO_ROOT}"
  exec caffeinate -dimsu conda run --no-capture-output -n ms2lda-hybrid \
    python -m benchmarks.fully_neural_ms2lda run \
    --run "${RUN_DIR}" \
    --source "${SOURCE_RUN}" \
    --reference "${REFERENCE_RUN}" >>"${LOG_FILE}" 2>&1
}

start_background() {
  mkdir -p "${LOG_DIR}"
  for command in screen caffeinate conda; do
    if ! command -v "${command}" >/dev/null 2>&1; then
      echo "Required command is unavailable: ${command}"
      exit 1
    fi
  done
  for directory in "${SOURCE_RUN}" "${REFERENCE_RUN}"; do
    if [[ ! -d "${directory}" ]]; then
      echo "Required input directory is unavailable: ${directory}"
      exit 1
    fi
  done
  if screen -list | grep -q "[.]${SESSION_NAME}"; then
    echo "Session ${SESSION_NAME} is already running."
    exit 1
  fi
  touch "${LOG_DIR}/launch.marker"
  screen -dmS "${SESSION_NAME}" /bin/bash "$0" foreground
  for _ in {1..30}; do
    if [[ -f "${RUN_DIR}/complete.json" ]]; then
      echo "Study is already complete. Log: ${LOG_FILE}"
      return
    fi
    if [[ "${RUN_DIR}/heartbeat.json" -nt "${LOG_DIR}/launch.marker" ]] && \
       screen -list | grep -q "[.]${SESSION_NAME}"; then
      echo "Started ${SESSION_NAME}. Log: ${LOG_FILE}"
      return
    fi
    sleep 1
  done
  echo "Runner did not publish a fresh heartbeat within 30 seconds."
  [[ -f "${LOG_FILE}" ]] && tail -n 80 "${LOG_FILE}"
  exit 1
}

case "${1:-status}" in
  start|resume)
    start_background
    ;;
  foreground)
    foreground
    ;;
  status)
    cd "${REPO_ROOT}"
    conda run --no-capture-output -n ms2lda-hybrid \
      python -m benchmarks.fully_neural_ms2lda status --run "${RUN_DIR}"
    screen -list | grep "[.]${SESSION_NAME}" || true
    ;;
  verify)
    cd "${REPO_ROOT}"
    conda run --no-capture-output -n ms2lda-hybrid \
      python -m benchmarks.fully_neural_ms2lda verify --run "${RUN_DIR}"
    ;;
  smoke)
    cd "${REPO_ROOT}"
    conda run --no-capture-output -n ms2lda-hybrid \
      python -m benchmarks.fully_neural_ms2lda smoke
    ;;
  *)
    echo "Usage: $0 {start|resume|status|verify|smoke}"
    exit 2
    ;;
esac

#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/joewandy/Work/git/MS2LDA"
DEFAULT_DATA_ROOT="/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs"
SOURCE_RUN="${NEURAL_ASSIGNMENT_SOURCE_RUN:-${DEFAULT_DATA_ROOT}/hybrid-lda-simplification-seed42-v8}"
REFERENCE_RUN="${NEURAL_ASSIGNMENT_REFERENCE_RUN:-${DEFAULT_DATA_ROOT}/indicative-msnlib-k1000-seed42-peak-pooling-correction}"
RUN_DIR="${NEURAL_ASSIGNMENT_RUN_DIR:-${DEFAULT_DATA_ROOT}/neural-assignment-ms2lda-seed42-v1}"
PROTOCOL_FILE="${NEURAL_ASSIGNMENT_PROTOCOL:-${REPO_ROOT}/benchmarks/neural_assignment_ms2lda/protocol.json}"
SESSION_NAME="${NEURAL_ASSIGNMENT_SESSION_NAME:-neural-assignment-ms2lda-seed42-v1}"
LAUNCH_WAIT_SECONDS="${NEURAL_ASSIGNMENT_LAUNCH_WAIT_SECONDS:-60}"
LOG_DIR="${RUN_DIR}/logs"
LOG_FILE="${LOG_DIR}/runner.log"
LAUNCH_MARKER="${LOG_DIR}/launch.marker"
FOREGROUND_MARKER="${LOG_DIR}/foreground.started"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export VECLIB_MAXIMUM_THREADS=4
export NUMEXPR_NUM_THREADS=4

preflight() {
  cd "${REPO_ROOT}"
  if [[ "$(git branch --show-current)" != "main" ]]; then
    echo "The long scientific run must start from merged fork main."
    exit 1
  fi
  if [[ -n "$(git status --porcelain)" ]]; then
    echo "The long scientific run requires a clean worktree."
    exit 1
  fi
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
  if [[ ! -f "${PROTOCOL_FILE}" ]]; then
    echo "Committed protocol is unavailable: ${PROTOCOL_FILE}"
    exit 1
  fi
  if [[ ! "${LAUNCH_WAIT_SECONDS}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Launch wait must be a positive integer number of seconds."
    exit 1
  fi
}

foreground() {
  mkdir -p "${LOG_DIR}"
  if [[ -n "${NEURAL_ASSIGNMENT_LAUNCH_TOKEN:-}" ]]; then
    printf '%s\n' "${NEURAL_ASSIGNMENT_LAUNCH_TOKEN}" >"${FOREGROUND_MARKER}"
  fi
  cd "${REPO_ROOT}"
  exec caffeinate -dimsu conda run --no-capture-output -n ms2lda-hybrid \
    python -m benchmarks.neural_assignment_ms2lda run \
    --run "${RUN_DIR}" \
    --source "${SOURCE_RUN}" \
    --reference "${REFERENCE_RUN}" \
    --protocol "${PROTOCOL_FILE}" >>"${LOG_FILE}" 2>&1
}

start_background() {
  preflight
  mkdir -p "${LOG_DIR}"
  if [[ -f "${RUN_DIR}/complete.json" ]]; then
    echo "Study is already complete. Log: ${LOG_FILE}"
    return
  fi
  if screen -list | grep -q "[.]${SESSION_NAME}"; then
    echo "Session ${SESSION_NAME} is already running."
    exit 1
  fi
  local launch_token
  launch_token="$(date -u +%Y%m%dT%H%M%SZ)-$$"
  printf '%s\n' "${launch_token}" >"${LAUNCH_MARKER}"
  screen -dmS "${SESSION_NAME}" env \
    NEURAL_ASSIGNMENT_LAUNCH_TOKEN="${launch_token}" \
    NEURAL_ASSIGNMENT_SOURCE_RUN="${SOURCE_RUN}" \
    NEURAL_ASSIGNMENT_REFERENCE_RUN="${REFERENCE_RUN}" \
    NEURAL_ASSIGNMENT_RUN_DIR="${RUN_DIR}" \
    NEURAL_ASSIGNMENT_PROTOCOL="${PROTOCOL_FILE}" \
    NEURAL_ASSIGNMENT_SESSION_NAME="${SESSION_NAME}" \
    NEURAL_ASSIGNMENT_LAUNCH_WAIT_SECONDS="${LAUNCH_WAIT_SECONDS}" \
    /bin/bash "$0" foreground
  for ((second = 0; second < LAUNCH_WAIT_SECONDS; second++)); do
    if [[ -f "${RUN_DIR}/complete.json" ]]; then
      echo "Study completed during launch. Log: ${LOG_FILE}"
      return
    fi
    if cmp -s "${LAUNCH_MARKER}" "${FOREGROUND_MARKER}" && \
       screen -list | grep -q "[.]${SESSION_NAME}"; then
      echo "Started ${SESSION_NAME}. Log: ${LOG_FILE}"
      return
    fi
    sleep 1
  done
  echo "Runner did not enter its foreground command within ${LAUNCH_WAIT_SECONDS} seconds."
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
      python -m benchmarks.neural_assignment_ms2lda status --run "${RUN_DIR}"
    screen -list | grep "[.]${SESSION_NAME}" || true
    ;;
  verify)
    cd "${REPO_ROOT}"
    conda run --no-capture-output -n ms2lda-hybrid \
      python -m benchmarks.neural_assignment_ms2lda verify --run "${RUN_DIR}"
    ;;
  smoke)
    cd "${REPO_ROOT}"
    conda run --no-capture-output -n ms2lda-hybrid \
      python -m benchmarks.neural_assignment_ms2lda smoke
    ;;
  *)
    echo "Usage: $0 {start|resume|status|verify|smoke}"
    exit 2
    ;;
esac

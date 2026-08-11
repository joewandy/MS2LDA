#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/Users/joewandy/Work/git/MS2LDA"
DATA_ROOT="/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs"

export NEURAL_ASSIGNMENT_RUN_DIR="${DATA_ROOT}/neural-assignment-ms2lda-seed42-v2-k1000-continuation"
export NEURAL_ASSIGNMENT_PROTOCOL="${REPO_ROOT}/benchmarks/neural_assignment_ms2lda/protocol_k1000_continuation.json"
export NEURAL_ASSIGNMENT_SESSION_NAME="neural-assignment-ms2lda-seed42-v2-k1000-continuation"

if [[ $# -eq 0 ]]; then
  set -- status
fi
exec "${REPO_ROOT}/scripts/run_neural_assignment_ms2lda.sh" "$@"

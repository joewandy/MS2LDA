#!/bin/zsh
set -eu
cd /Users/joewandy/Work/git/MS2LDA
run=/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42
data_root=/Users/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680
queue_log=/tmp/ms2lda_ecrtm_overnight_20260828.log
exec >> "$queue_log" 2>&1
lock=/tmp/ms2lda_ecrtm_overnight_20260828.lock
if ! mkdir "$lock" 2>/dev/null; then echo LOCK_EXISTS; exit 2; fi
trap 'rmdir "$lock" 2>/dev/null || true' EXIT
echo "QUEUE_START $(date)"
while kill -0 36455 2>/dev/null; do sleep 60; done
if [ ! -f "$run/models/ecrtm_canonical/result.json" ]; then
  echo "RESUME_TRAIN $(date)"
  conda run --no-capture-output -n ms2lda-neural python -m scripts.run_msnlib_model_comparison train-ecrtm-canonical --run "$run" --device cpu --epochs 40 --batch-size 200 --max-iter 1000
fi
if [ ! -f "$run/models/ecrtm_canonical/result.json" ]; then echo MISSING_RESULT_AFTER_TRAIN; exit 3; fi
if [ ! -f "$run/validation_chemical/ecrtm_canonical/complete.json" ]; then
  echo "CHEMICAL_RAW $(date)"
  conda run --no-capture-output -n ms2lda-msnlib-mag python -m scripts.run_msnlib_model_comparison chemical --run "$run" --data-root "$data_root" --method ecrtm_canonical
fi
if [ ! -f "$run/validation_chemical/ecrtm_canonical/complete.json" ]; then echo MISSING_RAW_CHEMISTRY; exit 4; fi
if [ ! -f "$run/validation_chemical/ecrtm_canonical_tau030/complete.json" ]; then
  echo "CHEMICAL_TAU030 $(date)"
  conda run --no-capture-output -n ms2lda-msnlib-mag python -m scripts.run_msnlib_model_comparison chemical --run "$run" --data-root "$data_root" --method ecrtm_canonical_tau030
fi
if [ ! -f "$run/validation_chemical/ecrtm_canonical_tau030/complete.json" ]; then echo MISSING_TAU030_CHEMISTRY; exit 5; fi
echo "QUEUE_DONE $(date)"

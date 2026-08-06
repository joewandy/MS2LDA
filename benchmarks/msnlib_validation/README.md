# Leakage-safe MSnLib validation

This package is the reproducible full-scale comparison of ordinary Tomotopy
LDA and the isolated ms2lda_hybrid research implementation. It does not change
the production MS2LDA backend.

The active correction uses all 38,888 eligible positive-mode spectra, 1,000
topics and seed 42. Complete Bemis–Murcko scaffold groups define train,
validation and test partitions; repeated compounds cannot cross partitions.
Vocabulary, pooled contextual features and learned transformations use
training spectra only. All held-out compounds are excluded from MAG.

## Evidence status

The earlier seed-42 Hybrid result is superseded. Its feature builder counted
document-level rounded words before locating a DreaMS peak, so separate
physical peaks that rounded to the same fragment or neutral-loss word inherited
one peak's contextual state. Tomotopy does not use this state and is unaffected.

indicative-msnlib-k1000-seed42-peak-pooling-correction.json is the active
post-review configuration. It explicitly records that earlier test results were
inspected. Hybrid features and models must be rebuilt from scratch; only the
unchanged Tomotopy core model may be imported through reuse-tomotopy.

The older configurations remain audit records:

- full-msnlib-k1000.json: historical seeds 42–46 design; it cannot now create
  pristine confirmatory evidence.
- indicative-msnlib-k1000-seed42.json: original laptop profile.
- indicative-msnlib-k1000-seed42-chemical-correction.json: superseded
  chemical-only correction whose Hybrid model used the flawed pooled features.

## Inputs and environments

Download and verify the positive-mode Zenodo 20179680 assets:

    DATA=/path/to/ms2lda-msnlib-validation/zenodo/20179680
    python scripts/download_msnlib_validation_assets.py --data-root "$DATA"
    python scripts/download_msnlib_validation_assets.py \
      --data-root "$DATA" --verify-only

Create both pinned environments. DreaMS is deliberately installed separately
with --no-deps because its metadata requests an obsolete PyTorch version;
environment-hybrid.yml supplies the validated patched runtime.

    conda env create -f environment-hybrid.yml
    conda run -n ms2lda-hybrid python -m pip install --no-deps \
      "git+https://github.com/pluskal-lab/DreaMS.git@dbec3a0b514a99e5056cfccde4559fda8cfe8129"

    conda env create -f environment-msnlib-mag.yml
    conda run -n ms2lda-msnlib-mag \
      python -m pip install --no-deps -e .

The existing MS2LDA_v2 environment is also valid for the historical MAG stack
and is used by the local corrective execution.

## Corrective run using the retained Tomotopy model

Start from a clean immutable source commit. The target run directory must not
exist or must be empty.

    REPO=/path/to/MS2LDA
    DATA=/path/to/ms2lda-msnlib-validation/zenodo/20179680
    SOURCE=/path/to/ms2lda-msnlib-validation/runs/indicative-msnlib-k1000-seed42/continuation/chemical-correction
    RUN=/path/to/ms2lda-msnlib-validation/runs/indicative-msnlib-k1000-seed42-peak-pooling-correction
    CONFIG=benchmarks/msnlib_validation/configs/indicative-msnlib-k1000-seed42-peak-pooling-correction.json

    cd "$REPO"
    conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
      validate-inputs --config "$CONFIG" --data-root "$DATA"
    conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
      preflight --config "$CONFIG" --data-root "$DATA"
    conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
      freeze-implementation-correction \
        --config "$CONFIG" --data-root "$DATA" --run "$RUN" \
        --repo-root "$PWD" --source-run "$SOURCE" \
        --reason "pool each physical peak into its own DreaMS contextual state"
    conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
      reuse-tomotopy --source-run "$SOURCE" --run "$RUN"

Launch the complete resumable pipeline without caffeinate. It uses four Hybrid
training threads, one inference thread and the frozen 250-epoch safety ceiling
with mandatory early stopping after five stable epochs.

    nohup conda run --no-capture-output -n ms2lda-hybrid \
      python -m benchmarks.msnlib_validation run-pipeline \
        --run "$RUN" --data-root "$DATA" --mag-environment MS2LDA_v2 \
      >> "$RUN/unattended.log" 2>&1 &

Reissuing run-pipeline is safe: each stage verifies completed artifacts and
resumes its own atomic checkpoints. If Hybrid reaches the safety ceiling
without satisfying both alpha and topic-word convergence, downstream chemical
evaluation and reporting stop.

## Fresh reconstruction

For a completely fresh run, clone through public HTTPS, check out the immutable
commit recorded in the final compact checkpoint, create the environments above,
then use ordinary freeze with the active configuration. Its immutable
evaluation_timing and prior_test_results_inspected fields prevent the result
from being mislabelled as prespecified. A fresh multicore Tomotopy fit is
scientifically equivalent but not bitwise identical to the retained model.

    git clone https://github.com/joewandy/MS2LDA.git
    cd MS2LDA
    git checkout <executed-source-commit-from-checkpoint>
    conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
      freeze --config "$CONFIG" --data-root "$DATA" --run "$RUN" \
      --repo-root "$PWD"
    conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
      run-pipeline --run "$RUN" --data-root "$DATA" \
      --mag-environment ms2lda-msnlib-mag

## Publication evidence and tests

After a real report completes, export the committed checkpoint and LaTeX
fragment from that report rather than transcribing values:

    conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
      export-publication --run "$RUN" \
      --checkpoint docs/hybrid_lda_seed42_checkpoint.json \
      --latex docs/hybrid_lda_seed42_results.tex \
      --manifest docs/hybrid_lda_seed42_publication_manifest.json

Large embeddings, models, indices and reports stay outside Git. The selected
handoff commits compact evidence and hashes; it does not claim a durable public
URI for the multi-gigabyte run.

Fast software validation:

    conda run -n ms2lda-hybrid \
      python -m benchmarks.msnlib_validation smoke

Smoke and synthetic outputs are software evidence only. Genuine chemical
evidence begins only when the peak-aware real-data report, full-spectrum SOS
and zero-leak MAG audit all complete. One seed cannot estimate cross-seed
stability, and the Zenodo assets contain no independent manual
motif–spectrum-annotation endpoint.

# Leakage-safe MSnLib validation

This package runs the full-scale comparison of ordinary Tomotopy LDA and the
isolated `ms2lda_hybrid` reference implementation. It is a benchmark surface,
not a production backend.

Two configurations are retained:

- `full-msnlib-k1000.json`: the deferred confirmatory design using all five
  prespecified seeds 42–46.
- `indicative-msnlib-k1000-seed42.json`: the practical full-data laptop run
  using seed 42 only. It is useful indicative evidence, not confirmation.

Both configurations use all 38,888 spectra that pass the published
preprocessing filters and all 1,000 topics. Molecules are assigned as complete
Bemis–Murcko scaffold groups, repeated compounds cannot cross partitions, and
all vocabulary and pooled-feature construction uses training spectra only.

HybridLDA treats the configured `alpha=0.6` as an initialization. During
classical topic discovery it independently estimates an asymmetric alpha from
training-document variational posteriors. Both lambda and alpha must converge
before encoder finalization or chemical evaluation. Initial and learned alpha
summaries are included in the machine-readable results.

The driver sequence is:

1. `validate-inputs` checks every external file against its frozen SHA-256.
2. `preflight` validates the full source and estimates required resources.
3. `freeze` writes immutable splits, completion masks, vocabulary, source
   hashes, dependency versions, and the protocol lock.
4. `run-core` builds the DreaMS cache and runs Tomotopy and HybridLDA.
5. `run-mag` uses `MS2LDA_v2` and excludes every validation/test compound
   from the MAG reference library before scoring.
6. `report` refuses incomplete or unconverged runs and writes
   machine-readable and manuscript-ready results.

The active single-seed run uses six Tomotopy workers, four Hybrid training
threads, one held-out inference thread, at most 50 Hybrid discovery epochs with
early stopping, and two rotating atomic checkpoints. Re-running `run-core`
resumes the newest valid checkpoint.

## Current full-data indicative run

```bash
DATA=/Users/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680
RUN=/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/indicative-msnlib-k1000-seed42
CONFIG=benchmarks/msnlib_validation/configs/indicative-msnlib-k1000-seed42.json

conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  validate-inputs --config "$CONFIG" --data-root "$DATA" \
  --output "$RUN/input_manifest.json"
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  preflight --config "$CONFIG" --data-root "$DATA" \
  --output "${RUN}.preflight.json"
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  freeze --config "$CONFIG" --data-root "$DATA" --run "$RUN" \
  --repo-root "$PWD" --test-results-inspected

export DATA RUN
caffeinate -dimsu /bin/zsh -lc '
  set -e
  cd /Users/joewandy/Work/git/MS2LDA
  conda run --no-capture-output -n ms2lda-hybrid \
    python -m benchmarks.msnlib_validation run-core --run "$RUN"
  conda run --no-capture-output -n MS2LDA_v2 \
    python -m benchmarks.msnlib_validation run-mag \
      --run "$RUN" --data-root "$DATA"
  conda run --no-capture-output -n ms2lda-hybrid \
    python -m benchmarks.msnlib_validation report --run "$RUN"
' > "$RUN/unattended.log" 2>&1
```

The lock deliberately records that earlier test results had already been
inspected before this scientific correction. No old result artifact is reused.

## Software smoke test

```bash
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation smoke
```

Smoke and synthetic outputs are software validation only. Genuine chemical
evidence begins only with a completed leakage-safe real-data report. A
single-seed report cannot estimate cross-seed stability, and the available
Zenodo assets contain no independent manual motif–spectrum annotation endpoint.
Tomotopy remains the production/reference backend.

# Leakage-safe MSnLib validation

This package runs the full-scale comparison of ordinary Tomotopy LDA and the
isolated `ms2lda_hybrid` reference implementation. MS2LDA and HybridLDA are
research software at this stage: this benchmark informs the preferred research
method, not a production migration or deployment decision.

Three configurations are retained:

- `full-msnlib-k1000.json`: the historical prespecified confirmatory design
  using seeds 42–46. It is retained for audit, but it cannot now produce
  pristine confirmatory evidence because the seed-42 test result is known.
- `indicative-msnlib-k1000-seed42.json`: the practical full-data laptop run
  using seed 42 only. It is useful indicative evidence, not confirmation.
- `indicative-msnlib-k1000-seed42-chemical-correction.json`: a post-hoc,
  name-only derivation that reuses the unchanged completed seed-42 models and
  corrects the chemical-evaluation representation and topic association.

All three configurations use all 38,888 spectra that pass the published
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
5. `run-chemical-inference` recomputes DreaMS and topic mixtures from every
   full held-out spectrum, without chemical labels. Document-completion halves
   remain exclusive to likelihood evaluation.
6. `run-mag` uses the historical `MS2LDA_v2` environment or the committed
   `ms2lda-msnlib-mag` reconstruction, excludes every validation/test compound
   from the MAG reference library, and requires a nonempty MAG-optimized motif
   before a topic can receive SOS.
7. `report` refuses incomplete or unconverged runs and writes
   machine-readable and manuscript-ready results.

The primary post-hoc SOS diagnostic assigns every test spectrum to its
highest-probability topic. This rank-based rule avoids an absolute probability
cutoff; it is not invariant to all calibration or prior differences. The
primary mean gives each held-out connectivity identity equal weight within its
topic, so replicate spectra do not act as independent chemical evidence; a
clearly labelled spectrum-weighted sensitivity is also saved. The
published `probability >= 0.5` association is retained unchanged as a
sensitivity analysis, not tuned after seeing Hybrid output. The driver reports
both SOS denominator formulas because the deposited paper notebook divides by
annotation bits whereas the supplement says to divide by the smaller
fingerprint. Its frozen MACCS/0.8 setting is identical between methods but is
not the downstream paper analysis notebook's RDKit/0.9 setting, so published
numerical bins are context rather than a comparable baseline.

## Acquire the public inputs from scratch

The benchmark needs only the positive-mode assets from Zenodo record 20179680.
The standard-library acquisition script resumes interrupted downloads, checks
the Zenodo MD5 and locally frozen SHA-256 digests, rejects unsafe ZIP members,
extracts through a staging directory, and validates every benchmark-facing
file before publishing the extraction:

```bash
DATA=/path/to/ms2lda-msnlib-validation/zenodo/20179680
python scripts/download_msnlib_validation_assets.py --data-root "$DATA"
python scripts/download_msnlib_validation_assets.py \
  --data-root "$DATA" --verify-only
```

Create the canonical Hybrid environment from `environment-hybrid.yml`. The
historical run used the existing local `MS2LDA_v2` environment for MAG; a
portable dependency-equivalent replacement is now recorded in
`environment-msnlib-mag.yml`:

```bash
conda env create -f environment-hybrid.yml
conda env create -f environment-msnlib-mag.yml
conda run -n ms2lda-msnlib-mag python -m pip install --no-deps -e .
```

The completed single-seed run uses six Tomotopy workers, four Hybrid training
threads, one held-out inference thread, and two rotating atomic checkpoints.
Its initial 50-epoch ceiling was reached with the topic-word matrix below the
frozen tolerance but training-only alpha still changing by 0.0141. No Hybrid
test result, MAG result, or report was produced at that point. The disclosed
continuation keeps every stopping criterion unchanged, raises only the safety
ceiling to 250 total epochs, and resumes the verified epoch-50 state. Early
stopping remains mandatory after five stable epochs.

The exact benchmark source used by the completed corrected run is committed at
`88b47d51b1f045e95810aeafafa042834d63372b`; it matches every entry in the
frozen `code_manifest.json`. A compact, shareable record of the protocol
lineage, split, leakage audit, SOS values, and large-artifact hashes is committed
as `docs/hybrid_lda_seed42_checkpoint.json`. See
`docs/hybrid_lda_method.tex` for the fresh end-to-end reconstruction command and
the explicit comparison with the Nature Communications method.

## Current full-data indicative run

```bash
REPO=/path/to/MS2LDA
DATA=/path/to/ms2lda-msnlib-validation/zenodo/20179680
SOURCE=/path/to/ms2lda-msnlib-validation/runs/indicative-msnlib-k1000-seed42
RUN="$SOURCE/continuation"
CONFIG=benchmarks/msnlib_validation/configs/indicative-msnlib-k1000-seed42.json

cd "$REPO"

conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  freeze-continuation --config "$CONFIG" --data-root "$DATA" --run "$RUN" \
  --repo-root "$PWD" --source-run "$SOURCE" \
  --reason "epoch 50 retained training-only alpha change above the frozen tolerance"
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  reuse-core-artifacts --source-run "$SOURCE" --run "$RUN"

export DATA RUN REPO
/bin/zsh -lc '
  set -e
  cd "$REPO"
  conda run --no-capture-output -n ms2lda-hybrid \
    python -m benchmarks.msnlib_validation run-core --run "$RUN"
  conda run --no-capture-output -n ms2lda-hybrid \
    python -m benchmarks.msnlib_validation _run-raw-dreams --run "$RUN"
  conda run --no-capture-output -n MS2LDA_v2 \
    python -m benchmarks.msnlib_validation run-mag \
      --run "$RUN" --data-root "$DATA"
  conda run --no-capture-output -n ms2lda-hybrid \
    python -m benchmarks.msnlib_validation report --run "$RUN"
' >> "$RUN/unattended.log" 2>&1
```

The continuation lock records that prior test results had been inspected, that
the trigger uses training state only, and that every setting except the maximum
epoch safety ceiling is unchanged. Feature data and the completed Tomotopy arm
are hash-verified before reuse. The two retained Hybrid checkpoints are
rebound to the continuation protocol without changing model, optimizer,
variational, RNG, convergence, or patience state.

## Corrected chemical endpoint

The original downstream SOS used the observed document-completion half of each
test spectrum and treated every clustered topic as annotated without applying
the paper's motif-optimization filter. Its numerical chemical output is
superseded. The corrected run retains the immutable original for audit, reuses
only hash-verified feature and core-model artifacts, and records that the
evaluation correction was designed after test results were inspected.

```bash
REPO=/path/to/MS2LDA
DATA=/path/to/ms2lda-msnlib-validation/zenodo/20179680
SOURCE=/path/to/ms2lda-msnlib-validation/runs/indicative-msnlib-k1000-seed42/continuation
RUN="$SOURCE/chemical-correction"
CONFIG=benchmarks/msnlib_validation/configs/indicative-msnlib-k1000-seed42-chemical-correction.json

cd "$REPO"

conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  freeze-chemical-correction --config "$CONFIG" --data-root "$DATA" \
  --run "$RUN" --repo-root "$PWD" --source-run "$SOURCE" \
  --reason "correct full-spectrum SOS, MAG optimization, rank-based association, and compound weighting after diagnosis"
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  reuse-core-artifacts --source-run "$SOURCE" --run "$RUN"
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  run-core --run "$RUN"
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  run-chemical-inference --run "$RUN"
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  _run-raw-dreams --run "$RUN"
conda run -n MS2LDA_v2 python -m benchmarks.msnlib_validation \
  run-mag --run "$RUN" --data-root "$DATA"
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation \
  report --run "$RUN"
```

## Software smoke test

```bash
conda run -n ms2lda-hybrid python -m benchmarks.msnlib_validation smoke
```

Smoke and synthetic outputs are software validation only. Genuine chemical
evidence begins only with a completed leakage-safe real-data report. A
single-seed report cannot estimate cross-seed stability, and the available
Zenodo assets contain no independent manual motif–spectrum annotation endpoint.
The corrected seed-42 run meets the publication-level one-seed SOS comparison
standard and supports HybridLDA as a credible preferred research method with
chemical coherence comparable to Tomotopy. It does not establish universal or
statistically resolved chemical superiority.

For a reviewable result snapshot without the large external artifacts, see
`docs/hybrid_lda_seed42_checkpoint.json`. It includes the exact source and
protocol identifiers, split and leakage counts, all four core inference arms,
latency, runtime, memory, topic diagnostics, raw-DreaMS results, and both SOS
definitions. The complete experimental setup, paper comparison, results, and
limitations are written in `docs/hybrid_lda_method.tex`.

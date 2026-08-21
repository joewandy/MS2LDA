# Neural MS2LDA research checkpoint

This directory contains one supported research model: a collapse-resistant,
fully neural hierarchical MS2LDA with 1,000 topics. A train-only co-occurrence
graph shapes topic words, a nearest-topic margin preserves distinct topics,
and a local-document product-of-experts router concentrates each spectrum in
one pass. It discovers topics without a Tomotopy teacher, DreaMS, variational
Bayes, chemistry labels, or test-set information. Tomotopy K=1000 is trained
separately as the established post-training comparator. The comparison reuses
a hash-verified six-worker Tomotopy training run and recomputes its held-out
inference with the same six-thread allowance as the neural model.

The complete resumable workflow is:

```text
download -> verify -> scaffold split -> first-seen training vocabulary
         -> train-only SGNS -> neural training -> Tomotopy comparison
         -> leakage-filtered MAG chemistry -> report
```

Run or resume it with:

```bash
python -m benchmarks.neural_assignment_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/run \
  --tomotopy-reference-run /path/to/frozen-tomotopy-run
```

Use `status --run ...` for a compact progress snapshot and `verify --run ...`
to recheck the immutable protocol, hashes, code provenance, and manifests. The
unattended shell wrapper in `scripts/run_neural_ms2lda.sh` pins the numerical
runtime to six CPU threads. Neural selection is fixed in advance to epoch 40;
validation measurements are diagnostics rather than acceptance gates.

The committed bundle and paper-style report are research artifacts. They do
not change the public MS2LDA application defaults, and the results are not
presented as Tomotopy parity.

Regenerate the paper figures and table from committed evidence with
`python scripts/generate_neural_ms2lda_report.py`. After compiling the LaTeX,
write or verify the report hash manifest with `--write-manifest` or
`--verify-manifest`, respectively.

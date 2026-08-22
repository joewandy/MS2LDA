# Neural MS2LDA research checkpoint

This directory contains one supported research model: a collapse-resistant,
fully neural hierarchical MS2LDA with 1,000 topics. A train-only co-occurrence
graph shapes topic words, a nearest-topic margin preserves distinct topics,
and a local-document product-of-experts router concentrates each spectrum in
one pass. A fixed document gate sharpens the final spectrum mixture, while a
soft token-type balance keeps fragments and neutral losses jointly represented
in topic words. Mean-normalized channel evidence removes fragment/loss
vocabulary-size bias before that balance. It discovers topics without a
Tomotopy teacher, DreaMS, variational Bayes, chemistry labels, or test-set
information. Tomotopy K=1000 is trained separately from the same frozen
training split as the established post-training comparator. Both methods use
the same six-thread allowance.

The complete resumable workflow is:

```text
download -> verify -> scaffold split -> first-seen training vocabulary
         -> train-only SGNS -> neural training -> Tomotopy comparison
         -> leakage-filtered MAG chemistry -> report
```

Create the two pinned environments and acquire the checksum-locked public data
(about 3.6 GB of archives plus extracted files) with:

```bash
conda env create -f environment-neural-ms2lda.yml
conda env create -f environment-msnlib-mag.yml
conda run -n ms2lda-neural python scripts/download_msnlib_validation_assets.py \
  --data-root /path/to/MSnLib-assets
```

Run or resume the complete workflow directly with:

```bash
python -m benchmarks.neural_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/run
```

Use `status --run ...` for a compact progress snapshot and `verify --run ...`
to recheck the immutable protocol, hashes, code provenance, and manifests. The
module entry point pins the numerical runtime to six CPU threads before
importing numerical libraries. Neural selection is fixed in advance to epoch
40; the workflow completes both validation evaluations before opening the test
split.

The committed bundle and paper-style report are research artifacts. They do
not change the public MS2LDA application defaults. The single seed-42 result is
not multi-seed confirmation or evidence to replace the production backend.

Regenerate the paper figures and table from committed evidence with
`python scripts/generate_neural_ms2lda_report.py`. After compiling the LaTeX,
write or verify the report hash manifest with `--write-manifest` or
`--verify-manifest`, respectively.

# Neural MS2LDA study

This directory contains one seed-42, K=1000 research model with one-pass
document inference. Train-only 48-dimensional SGNS features and fragment/loss
indicators are projected into a shared topic geometry. A linear leave-one-out
context map and whole-spectrum evidence define top-2 token routes; their
count-weighted aggregation produces each document mixture. The decoder assigns
fixed equal mass to fragment and neutral-loss channels.

Training uses full spectra and alternates router and topic updates. Sinkhorn
targets, a positive-NPMI graph, prototype separation, and routing-temperature
annealing protect the topic inventory. PyTorch deterministic algorithms are
required under the six-thread allowance. The validation-only ablation ledger
is `results/seed42/ablation_results.json`.

Tomotopy is an independently trained comparator, not a teacher. Both methods
use six CPU threads. This is a single-dataset, single-seed research comparison;
it does not change the production MS2LDA backend.

Create the two environments and acquire the public inputs:

```bash
conda env create -f environment-neural-ms2lda.yml
conda env create -f environment-msnlib-mag.yml
conda run -n ms2lda-neural python scripts/download_msnlib_validation_assets.py \
  --data-root /path/to/MSnLib-assets
```

Run the complete reproducibility workflow:

```bash
conda run -n ms2lda-neural python -m benchmarks.neural_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/run
```

`status --run /path/to/run` prints progress. The published model artifact
contains only `weights.pt`, `model.json`, and `vocabulary.json`. `results.json`
is the sole numerical source for the report. Regenerate its fragments with:

```bash
python scripts/generate_neural_ms2lda_report.py
```

See [HANDOVER.md](HANDOVER.md) for the exact architecture, evidence contract,
and verification commands.

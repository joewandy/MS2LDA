# Neural MS2LDA study

This directory contains one research model: a collapse-resistant neural
MS2LDA with 1,000 topics and one-pass document inference. Fixed train-only token
features and learned topic prototypes define both the contextual top-2 router
and the fragment/loss-balanced decoder. Paired views, balanced assignments, a
positive-NPMI graph, prototype separation, and deterministic dead-topic
recycling protect the large topic inventory during training.

Tomotopy is trained independently from the same training split as the
comparator. Both methods use six CPU threads. Validation evaluation and MAG
scoring finish before the workflow opens the test matrices.

Create the two environments and acquire the public inputs:

```bash
conda env create -f environment-neural-ms2lda.yml
conda env create -f environment-msnlib-mag.yml
conda run -n ms2lda-neural python scripts/download_msnlib_validation_assets.py \
  --data-root /path/to/MSnLib-assets
```

Run the study:

```bash
conda run -n ms2lda-neural python -m benchmarks.neural_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/run
```

`status --run /path/to/run` prints current progress. An interrupted fit restarts
from its deterministic initialization.

The published model artifact contains only `weights.pt`, `model.json`, and
`vocabulary.json`. `results.json` is the sole numerical source for the report.
Regenerate its figures, tables, and prose macros with:

```bash
python scripts/generate_neural_ms2lda_report.py
```

This is a single-dataset, single-seed research result. It does not change the
public MS2LDA application defaults or establish that the neural model should
replace the production Tomotopy backend.

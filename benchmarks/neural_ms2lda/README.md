# Neural MS2LDA study

This directory contains the locked seed-42, K=1000 Neural MS2LDA M1 model with
one-pass document inference. Each token has 48 train-only SGNS coordinates and
two fragment/loss indicators. A bias-free projection maps those 50 features to
128 dimensions, and a nonlinear context router combines each token with the
leave-one-out context of its spectrum:

`Linear(256, 128, bias=False) -> GELU`.

The routed tokens interact with 1,000 learned motif prototypes. Sparse top-2
assignments, additive whole-spectrum evidence, and a detached document gate
produce the document mixture. The decoder gives fragment and neutral-loss
channels equal probability mass.

Training uses full spectra and alternates router and topic updates. Sinkhorn
targets, positive-NPMI structure, prototype separation, routing-temperature
annealing, finite checks, and gradient clipping protect the topic inventory.
The model has 167,168 learned parameters.

## Why this model

The internal simplification campaign selected M1 over larger and lower-dimensional
router variants. Its deterministic lock produced 884 optimized, 408 evaluable,
and 265 useful validation motifs with mean SOS 0.6580793714 and validation NLL
8.9741399256. Exact internal ablations remain in
`results/seed42/ablation_results.json` and `SIMPLIFICATION.md`.

A later validation-only external comparison tested canonical fixed-SGNS ETM, a
deterministic pooled projected model, fragment/loss-balanced ETM, and canonical
ECRTM. None preserved the complete M1 chemistry, completion, and topic-inventory
contract. The most revealing failure was a 614-topic near-exact duplicate
component in the pooled model. Channel balancing repaired ETM annotation
coverage but not evaluable/useful breadth or SOS. The maintained ECRTM Sinkhorn
path failed its convergence contract at real K/V after 21 completed epochs.

M1 is therefore the **least-complex model demonstrated to satisfy the complete
real-data scientific contract**. It is not claimed to be the simplest
conceivable model or a universally superior topic model. See
[FINAL_MODEL_SELECTION.md](FINAL_MODEL_SELECTION.md) for the locked decision.
No alternative candidate is authorized for test.

Tomotopy is an independently trained comparator, not a teacher. This remains a
single-dataset, single-data-split research result and does not change the
production MS2LDA backend.

## Required diagnostics

Neural evaluation now reports more than likelihood and MAG coverage. The frozen
contract in `diagnostics.py` includes mixture sparsity, active and unique top-1
topics, duplicate beta components at cosine 0.95/0.99/0.999, beta concentration,
top-word uniqueness, and fragment/loss probability mass. A large optimized
motif count alone is not evidence of a usable topic inventory.

For an existing validation run, backfill the new diagnostics without opening
test data:

```bash
python scripts/backfill_neural_ms2lda_diagnostics.py --run /path/to/run
```

## Reproduction

Create the unified production, neural, CUDA, and MAG environment and acquire
the public inputs:

```bash
conda env create -f environment.yml
conda run -n ms2lda-neural python scripts/download_msnlib_validation_assets.py \
  --data-root /path/to/MSnLib-assets
```

`environment.yml` is the sole Conda manifest. The active pipeline runs the
MS2LDA application, training, validation, FAISS/Spec2Vec retrieval, and MAG
annotation with its `ms2lda-neural` interpreter.

Run the complete locked M1/Tomotopy workflow:

```bash
conda run -n ms2lda-neural python -m benchmarks.neural_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/run
```

`status --run /path/to/run` prints progress. The published model artifact
contains only `weights.pt`, `model.json`, and `vocabulary.json`.

Regenerate the original numerical fragments and the external model-selection
tables with:

```bash
python scripts/generate_neural_ms2lda_report.py
python scripts/generate_neural_ms2lda_model_selection.py
```

See [HANDOVER.md](HANDOVER.md) for the exact architecture, evidence contract,
and verification commands. The next compute campaign is the validation-only M1
optimization-seed study specified in
`research/etm_ecrtm_msnlib/M1_MULTISEED_HANDOFF.md`.

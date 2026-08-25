# Neural MS2LDA study

This directory contains one seed-42, K=1000 deep neural topic model with
one-pass document inference. Each token has 48 train-only SGNS coordinates and
two fragment/loss indicators. A bias-free projection maps those 50 features to
128 dimensions, and a nonlinear context router combines each token with the
leave-one-out context of its spectrum:

`Linear(256, 256) -> GELU -> LayerNorm(256) -> Linear(256, 128)`.

The routed tokens interact with 1,000 learned motif prototypes. Sparse top-2
assignments, additive whole-spectrum evidence, and a detached document gate
produce the document mixture. The decoder gives fragment and neutral-loss
channels equal probability mass.

Training uses full spectra and alternates router and topic updates. Sinkhorn
targets, positive-NPMI structure, prototype separation, routing-temperature
annealing, finite checks, and gradient clipping protect the topic inventory.
The model has 233,600 learned parameters.

## Why this model

The selected U1 formulation removes Fourier mass coordinates, partial training
views, Jensen--Shannon view consistency, local reconstruction, dead-topic
recycling, weighted k-means++ initialization, and adaptive channel mass. Its
deterministic lock produced 843 optimized, 429 evaluable, and 268 useful
validation motifs with mean SOS 0.6506700670 and validation NLL 8.8320026353.

An earlier scratch reconstruction narrowly missed two thresholds derived from a
historical U1 run. The provenance-grounded deterministic lock passed every
historical U1 reconstruction threshold and substantially improved the accepted
control's motif inventory while removing seven auxiliary mechanisms. Under the
project's practical-tie rule, the simpler deep model is selected and the worse
predictive NLL remains an explicit trade-off. Exact values and the earlier
mechanical misses remain in `results/seed42/ablation_results.json`.

The nonlinear router is a required architectural invariant. Shallow U7 and
S-series endpoints are ablation evidence only. A future DreaMS embedding can
augment or replace the pooled spectrum-context vector before the nonlinear
router; DreaMS is not integrated in the reported model.

Tomotopy is an independently trained comparator, not a teacher. Its committed
evidence is fixed and this method-development update does not rerun it. This is
a single-dataset, single-seed research result and does not change the production
MS2LDA backend.

## Reproduction

Create the two environments and acquire the public inputs:

```bash
conda env create -f environment-neural-ms2lda.yml
conda env create -f environment-msnlib-mag.yml
conda run -n ms2lda-neural python scripts/download_msnlib_validation_assets.py \
  --data-root /path/to/MSnLib-assets
```

Run the complete workflow:

```bash
conda run -n ms2lda-neural python -m benchmarks.neural_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/run
```

`status --run /path/to/run` prints progress. The published model artifact
contains only `weights.pt`, `model.json`, and `vocabulary.json`. `results.json`
is the sole paper-facing numerical source. Regenerate report fragments with:

```bash
python scripts/generate_neural_ms2lda_report.py
```

See [HANDOVER.md](HANDOVER.md) for the exact architecture, evidence contract,
and verification commands. See [SIMPLIFICATION.md](SIMPLIFICATION.md) for the
measured removals and selection rationale.

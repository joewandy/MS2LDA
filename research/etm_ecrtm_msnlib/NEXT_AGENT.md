# Principled sparse-ETM next-agent handoff

## Mission

Run a bounded, validation-only campaign to test whether principled sparsity can
give ETM M1-like high-confidence spectrum assignments without restoring M1's
bespoke routing stack.

The permitted mechanism families are:

1. sparsemax or entmax document-topic mixtures;
2. a principled sparse prior;
3. pseudo-count objective scaling so reconstruction mass does not overwhelm
   the sparse prior or regularizer.

Do not combine all mechanisms immediately. Isolate them in a predeclared,
bounded screen so any improvement has a clear explanation.

## Required sequence

### 1. Truth-known synthetic screen

Start with the existing MS/MS simulator and its frozen seeds, fragment/loss
representation, planted motifs, pseudo-count construction, and recovery
metrics. Compare against the existing fixed-SGNS ETM and relevant dense
baselines. Report topic recovery, document-mixture recovery, effective topics
per spectrum, topic occupancy/redundancy, completion NLL, and stability.

Do not use real validation chemistry to choose the sparse formulation. Advance
only a clearly justified synthetic candidate; do not conduct an open-ended
hyperparameter search.

### 2. Real MSnLib validation only if warranted

If the synthetic screen supports a candidate, freeze its form and evaluate it
on the existing seed-42 train/validation artifacts. Reuse the train-only
vocabulary, SGNS features, leakage-filtered MAG index, annotation machinery,
membership threshold, frozen chemistry gates, and diagnostics unchanged.

Candidate test data remain locked. Do not load, inspect, score, or summarize
candidate test artifacts.

### 3. Fallback

If the bounded sparse-ETM campaign fails, stop that path. M1 multiseed stability
then becomes the fallback campaign described in `M1_MULTISEED_HANDOFF.md`.

## Shared artifacts to preserve and reuse

- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/neural-minimality-seed42/m1-lock`

These contain the raw MSnLib input, prepared train/validation matrices,
vocabulary, SGNS features, Spec2Vec model/database, filtered FAISS index, and
baseline M1/ETM evidence. Treat them as immutable shared inputs and verify their
recorded hashes before use.

## Environment and guardrails

Use the repository's sole Conda manifest, `environment.yml`, and the existing
`ms2lda-neural` environment. CUDA PyTorch is validated on the RTX 5070, and the
real validation-only MAG load/query/annotation smoke has passed.

Do not add separation, NPMI, Sinkhorn, token routing, another detached-gate run,
or another architecture family during this campaign. Do not overwrite baseline
or shared artifacts, and do not commit large arrays, model weights, databases,
or indexes.

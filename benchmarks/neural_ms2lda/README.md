# Contextual Sparse ETM study

This directory contains the scientific implementation and tests for
**Contextual Sparse ETM**, an Embedded Topic Model (ETM) adapted to short
tandem-mass-spectrometry documents.

The model retains the published ETM generator, embedding-space topic--word
decoder, Gaussian variational posterior, multinomial reconstruction term and
analytic Gaussian KL divergence. It makes three explicit adaptations:

1. fragment and neutral-loss decoder channels each receive half of every
   topic's probability mass;
2. leave-one-out token context contributes top-2 evidence to the ETM posterior
   mean through one learned scalar; and
3. published 1.5-entmax replaces posterior softmax to produce exact zeros in
   each spectrum's topic mixture.

The only learned parameter added to the channel-balanced ETM is the context
scalar. The model-specific mathematics is implemented as named tensor
functions in `contextual_sparse_etm.py`; the sole `nn.Module` is a thin
parameter and checkpoint shell. Normalized count input and the raw pseudo-count
reconstruction equation live in `topic_model_training.py`.

## Final held-out test comparison

All neural rows use the same train-only vocabulary and SGNS coordinates,
K=1,000, training seed 7043, 120 epochs and CUDA execution. Models were fitted
on train, developed on validation, frozen, and evaluated once on the fixed test
split.

| Model | Optimized | Evaluable | Useful | Mean SOS | Median SOS | Completion NLL |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical ETM | 601 | 171 | 101 | 0.631759 | 0.633333 | **8.686003** |
| Balanced ETM | **887** | 207 | 130 | 0.644232 | 0.640351 | 8.779686 |
| **Contextual Sparse ETM** | 799 | **572** | **343** | 0.637702 | 0.639500 | 9.535540 |
| Tomotopy LDA | 609 | 319 | 188 | **0.652752** | **0.651515** | 9.739090 |

The result is a breadth--quality trade-off, not uniform dominance. Contextual
Sparse ETM has the broadest evaluable and useful motif inventory and sparse
per-spectrum mixtures. Dense ETM controls retain better completion NLL, while
Tomotopy retains the highest conditional SOS over a smaller evaluable set.

Across training seeds 7043, 23 and 37, Contextual Sparse ETM yields 557--582
evaluable and 327--353 useful test motifs, median 3.68--3.74 effective topics
per spectrum and 914--922 unique top-1 topics. Every run is finite, has zero MAG
clustering or optimization exceptions, and avoids a catastrophic duplicate
component.

## Canonical files

- `contextual_sparse_etm.py` -- decoder, contextual evidence, posterior, KL and
  entmax equations.
- `topic_model_training.py` -- normalized count input and sparse raw-count
  reconstruction.
- `reproduction_plan.py` -- ordered 54-stage clean-room protocol.
- `reproduction_audit.py` -- chronology, split-boundary, hash and probability
  checks.
- `tests/test_contextual_sparse_etm.py` -- equation-level and deterministic
  inference correspondence.
- `tests/test_reproduction_evidence.py` -- evidence and manuscript-claim gates.
- `FINAL_MODEL_SELECTION.md` -- final scientific decision and interpretation.
- `HANDOVER.md` -- exact evidence locations and continuation rules.

The complete scientific report is
`docs/research/neural_ms2lda_report.tex`, with its reviewed PDF beside it.

## Verification

From the repository root, in the recorded reproduction environment:

```bash
pytest -q benchmarks/neural_ms2lda/tests
black --check \
  benchmarks/neural_ms2lda/contextual_sparse_etm.py \
  benchmarks/neural_ms2lda/model_evaluation.py \
  benchmarks/neural_ms2lda/reproducibility.py \
  benchmarks/neural_ms2lda/reproduction_audit.py \
  benchmarks/neural_ms2lda/reproduction_plan.py \
  benchmarks/neural_ms2lda/topic_model_training.py \
  scripts/run_contextual_sparse_etm.py \
  scripts/run_contextual_sparse_etm_reproduction.py \
  scripts/package_contextual_sparse_etm_reproduction.py \
  scripts/generate_routing_etm_report.py
ruff check \
  benchmarks/neural_ms2lda/contextual_sparse_etm.py \
  benchmarks/neural_ms2lda/model_evaluation.py \
  benchmarks/neural_ms2lda/reproducibility.py \
  benchmarks/neural_ms2lda/reproduction_audit.py \
  benchmarks/neural_ms2lda/reproduction_plan.py \
  benchmarks/neural_ms2lda/topic_model_training.py \
  scripts/run_contextual_sparse_etm.py \
  scripts/run_contextual_sparse_etm_reproduction.py \
  scripts/package_contextual_sparse_etm_reproduction.py \
  scripts/generate_routing_etm_report.py
```

The packaged evidence and its machine-verifiable acceptance status are under
`research/etm_ecrtm_msnlib/local_results/20260901_contextual_sparse_etm_reproduction/`.

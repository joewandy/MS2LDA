# Neural MS2LDA handover

## Current publication direction (2026-08-30)

M1 is retained as a private benchmark and source of ablation evidence, not as
the proposed publication model. The active direction is a published balanced
ETM base with top-2 contextual posterior evidence and entmax 1.5. Its validation
result is 803 optimized, 445 evaluable and 289 useful motifs, mean SOS 0.647153,
completion NLL 9.542924, median 3.70 effective topics and 828 unique top-1
topics. It exceeds M1 and Tomotopy on evaluable/useful discovery breadth and is
now frozen as the paper-facing validation baseline. It does not pass the complete
historical gate Boolean, so test stays locked. Three unchanged Routing ETM
training seeds now reproduce both the discovery advantage and the residual
trade-off; M1 multiseed is not the next task. See
`research/etm_ecrtm_msnlib/NEXT_AGENT.md`.

## Current state

The repository supports one locked seed-42, K=1000, 40-epoch Neural MS2LDA
architecture under a six-CPU-thread allowance. Its deterministic validation
result is 884 optimized motifs, 408 high-confidence evaluable motifs, 265 useful
motifs, mean SOS 0.6580793714, median SOS 0.6488636364, and completion NLL
8.9741399256.

M1 first survived a bounded within-family simplification campaign. A later
validation-only external campaign compared it with canonical fixed-SGNS ETM,
a deterministic pooled projected model, fragment/loss-balanced ETM, and
canonical ECRTM. No completed alternative preserved the full chemistry,
completion, and inventory contract; ECRTM failed its maintained Sinkhorn
convergence path at real K/V before evaluation.

The correct conclusion is therefore that M1 is the **least-complex model
demonstrated to satisfy the complete real-data scientific contract**. It is not
the simplest conceivable architecture, and the study does not establish a
production-backend replacement. Alternative candidate test data remain locked.
See `FINAL_MODEL_SELECTION.md` for the complete decision.

## Implemented model

1. Train 48-dimensional SGNS token coordinates on training spectra only. Append
   two fragment/loss indicators and normalize the resulting 50-dimensional
   token features.
2. Apply a bias-free 50-to-128 projection. Select 1,000 distinct prototype
   starting tokens with the seed-42 uniform permutation and initialize learned
   128-dimensional prototypes from those projected tokens.
3. Concatenate each projected token with its count-weighted leave-one-out
   spectrum context. Apply one bias-free `Linear(256, 128)` followed by GELU,
   add the nonlinear correction to the token vector, and normalize.
4. Add whole-spectrum prototype evidence, retain and renormalize the top two
   token assignments, and aggregate their count-weighted mass.
5. Multiply routed topic mass by detached whole-spectrum evidence raised to
   0.75. Exact zero support is preserved and empty spectra receive a uniform
   mixture.
6. Decode with cosine logits and separate fragment/loss softmaxes. Each channel
   receives exactly half of the topic probability mass.
7. Train on full spectra with alternating router and topic blocks. Retain
   Sinkhorn balancing, positive-NPMI regularization, prototype separation,
   routing-temperature annealing, deterministic execution, finite checks, and
   gradient clipping.

The model has 167,168 learned parameters: 6,400 in the token projection, 32,768
in the nonlinear context router, and 128,000 in the topic prototypes. The
production-facing M1 implementation remains unconditional: there are no
architecture flags or loader compatibility branches.

## Why the retained mechanisms are functional

Internal M1 ablations showed substantial motif losses when the document gate,
Sinkhorn, NPMI, or prototype separation was removed. The external comparison
then reproduced the corresponding failure modes in simpler model families:

- pooled projected MS2LDA formed a 614-topic near-exact duplicate component;
- ETM and balanced ETM produced diffuse/weak spectrum-topic assignments;
- balancing ETM's channels improved optimized coverage but not chemical breadth
  or SOS;
- ECRTM targeted collapse but its maintained ordinary-domain Sinkhorn path
  became numerically and operationally unsuitable at K=1000, V=21,233.

These results do not prove that every future alternative must fail. They show
that the present M1 mechanisms address observed real-data defects rather than
adding complexity without evidence.

## Removed mechanisms

The final M1 implementation contains no Fourier mass coordinates, paired
partial views, Jensen--Shannon view consistency, local reconstruction,
dead-topic recycling, weighted k-means++ initialization, or adaptive
fragment/loss channel mass. Their exact ablation measurements remain in the
single ledger; their code paths and protocol fields do not.

Shallow U7 and S-series endpoints are excluded because they remove the
nonlinear learned representation block. A future DreaMS embedding may augment
or replace pooled context, but it is not part of this model-selection claim.

## Expanded diagnostic contract

Likelihood and optimized motif count are not sufficient. Every new neural
model result must report the contract implemented in `diagnostics.py`:

- median/mean effective topics per spectrum and corpus effective topics;
- active topics, maximum mean usage, unique top-1 topics, and never-top-1 topics;
- mean/median nearest beta cosine and maximum pairwise cosine;
- duplicate connected components at cosine 0.95, 0.99, and 0.999;
- median beta effective words, maximum probability, and top-20 mass;
- top-word uniqueness;
- per-topic fragment probability mass and extreme-skew fraction.

The default catastrophic-component flag is a connected component containing at
least half the topics at the strictest 0.999 cosine threshold. Chemistry gates
remain separate and primary.

## Authoritative evidence

| Purpose | Location |
| --- | --- |
| Canonical combined report | `docs/research/neural_ms2lda_report.tex` |
| Archived pre-selection methods report | `docs/research/archive/neural_ms2lda_report_pre_model_selection.tex` |
| Final external selection decision | `benchmarks/neural_ms2lda/FINAL_MODEL_SELECTION.md` |
| Fixed study constants and diagnostic settings | `benchmarks/neural_ms2lda/protocol.json` |
| Model and one-pass inference | `benchmarks/neural_ms2lda/model.py` |
| Training objectives | `benchmarks/neural_ms2lda/objectives.py` |
| Alternating updates | `benchmarks/neural_ms2lda/training.py`, `optimization.py` |
| Data and train-only token features | `benchmarks/neural_ms2lda/data.py`, `spectra.py` |
| Inventory diagnostics | `benchmarks/neural_ms2lda/diagnostics.py` |
| Canonical model | `benchmarks/neural_ms2lda/results/seed42/trained_model/` |
| Locked M1/Tomotopy evidence | `benchmarks/neural_ms2lda/results/seed42/results.json` |
| Ablation ledger | `benchmarks/neural_ms2lda/results/seed42/ablation_results.json` |
| External validation comparison | `research/etm_ecrtm_msnlib/local_results/20260827_followup/comparison.csv` |
| External experiment log | `research/etm_ecrtm_msnlib/local_results/20260827_followup/EXPERIMENT_LOG.md` |
| Historical M1 stability plan (not current) | `research/etm_ecrtm_msnlib/M1_MULTISEED_HANDOFF.md` |
| Current routing-informed ETM evidence | `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/README.md` |
| Frozen Routing ETM manifest | `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/checkpoint_manifest.json` |
| Routing ETM integrity checker | `scripts/verify_routing_etm_checkpoint.py` |
| Current next campaign | `research/etm_ecrtm_msnlib/NEXT_AGENT.md` |

`results.json` remains the sole numerical source for the locked M1/Tomotopy
paper comparison. The separate committed `comparison.csv` is the numerical
source for the later validation-only external model-selection section.

## Reproduction

```bash
conda env create -f environment.yml

conda run -n ms2lda-neural python \
  scripts/download_msnlib_validation_assets.py \
  --data-root /path/to/MSnLib-assets

conda run -n ms2lda-neural python -m benchmarks.neural_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/new-run
```

This unified environment includes the production MS2LDA application, neural
and CUDA dependencies, and the FAISS/Spec2Vec MAG annotation stack.

Backfill the expanded diagnostics for a completed validation run without
opening test:

```bash
python scripts/backfill_neural_ms2lda_diagnostics.py --run /path/to/run
```

Regenerate report inputs:

```bash
python scripts/generate_neural_ms2lda_report.py
python scripts/generate_neural_ms2lda_model_selection.py
```

## Verification

```bash
black --check benchmarks/neural_ms2lda \
  scripts/download_msnlib_validation_assets.py \
  scripts/generate_neural_ms2lda_report.py \
  scripts/generate_neural_ms2lda_model_selection.py \
  scripts/backfill_neural_ms2lda_diagnostics.py

ruff check --select E,F,I benchmarks/neural_ms2lda \
  scripts/download_msnlib_validation_assets.py \
  scripts/generate_neural_ms2lda_report.py \
  scripts/generate_neural_ms2lda_model_selection.py \
  scripts/backfill_neural_ms2lda_diagnostics.py

pytest -q benchmarks/neural_ms2lda/tests
python scripts/generate_neural_ms2lda_report.py
python scripts/generate_neural_ms2lda_model_selection.py
```

Also run the production regression suite with the documented frozen-upstream
exclusions, inspect the built wheel to confirm research code is excluded, import
the installed wheel outside the checkout, compile the LaTeX report
deterministically, and visually inspect every rendered PDF page.

## Next compute

The unchanged Routing ETM has now been verified on three real training seeds over
the same frozen validation split. Every run preserves higher evaluable/useful
breadth than M1 and Tomotopy, sparse mixtures and a broad inventory; every run
also reproduces the coverage/SOS/NLL deficit relative to M1. Stop repeating
identical seeds. If more model work is wanted, positive-NPMI is the one optional
bounded intervention now justified by that residual. Do not start M1 multiseed;
candidate test remains locked. The current workflow is in
`research/etm_ecrtm_msnlib/NEXT_AGENT.md`.

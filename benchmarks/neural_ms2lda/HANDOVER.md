# Contextual Sparse ETM handover

## Current state

The model and evaluation protocol are frozen. A complete clean-room run has
finished from public asset acquisition through train/validation/test splitting,
train-only vocabulary and SGNS construction, Tomotopy, two ETM controls,
synthetic component studies, three Contextual Sparse ETM training seeds, frozen
test evaluation, MAG/SOS scoring, evidence packaging and report generation.

- Reproduction ID: `c8e26d27-861e-42ed-a04f-ca5a9ecfedae`
- Raw-run source commit: `379ab80ab6f0916e0c31a036bab229dda1d4727a`
- Raw result root:
  `/home/joewandy/Work/data/MS2LDA-reproductions/20260901-379ab80-gpu-clean`
- Committed compact evidence:
  `research/etm_ecrtm_msnlib/local_results/20260901_contextual_sparse_etm_reproduction`
- Report source and PDF: `docs/research/neural_ms2lda_report.tex` and
  `docs/research/neural_ms2lda_report.pdf`

The raw source worktree was clean. All 54 planned stages completed in order and
their owned outputs still match their recorded SHA-256 hashes. The compact
packager independently rechecks those hashes, model-view identities, release
chronology, probability matrices, chemical-accounting invariants and scientific
claim gates before producing report inputs.

## Data and evaluation boundary

The public source contains 41,568 spectra; 38,888 pass preprocessing. A
deterministic group-disjoint 70/10/20 split with seed 42 gives:

| Split | Spectra |
| --- | ---: |
| Train | 27,222 |
| Validation | 3,889 |
| Test | 7,777 |

The 21,233-word vocabulary and 48-dimensional SGNS coordinates use training
spectra only. There are zero leaked connectivity keys or split groups. The MAG
reference index removes every molecule whose connectivity key occurs in the
held-out split being scored.

Models are fitted on train and developed on validation. Each model checkpoint
and every required validation output is hashed before the test view is exposed.
Test evaluation performs no fitting, selection or spectrum-specific
optimization.

## Final model

Contextual Sparse ETM is a published ETM backbone with:

- fixed train-only SGNS word coordinates;
- a channel-balanced ETM decoder with 50/50 fragment/loss probability mass;
- a two-layer 800-unit ETM encoder and diagonal-Gaussian posterior;
- leave-one-out contextual top-2 evidence, added as a centred log offset to the
  posterior mean through one learned scalar;
- 1.5-entmax document-topic mixtures; and
- raw pseudo-count multinomial reconstruction plus analytic Gaussian KL.

At deterministic inference, the latent vector equals the shifted posterior
mean and is mapped through the same 1.5-entmax equation used during training.
The implementation has no architecture switch or spectrum-level optimizer.

## Evidence summary

The primary held-out test run has 572 evaluable and 343 useful motifs, median
3.680 effective topics, median exact support 6, 917 unique top-1 topics and
completion NLL 9.535540. Its mean/median SOS are 0.637702/0.639500. Across the
three initialization seeds, evaluable motifs range from 557 to 582 and useful
motifs from 327 to 353.

The truth-known K=128 stress experiment recovers all 18 planted motifs at
matched topic-word cosine at least 0.50. This recovery count is distinct from
the 19 fitted topics that become a top-1 winner for at least one spectrum.

## Reproduce or verify

For a new clean run, choose new `RUN` and `EVIDENCE` directories:

```bash
python -m scripts.run_contextual_sparse_etm_reproduction initialize --root RUN
python -m scripts.run_contextual_sparse_etm_reproduction run --root RUN
python -m scripts.package_contextual_sparse_etm_reproduction \
  --root RUN --output EVIDENCE
python -m scripts.generate_routing_etm_report \
  --evidence-root EVIDENCE --output docs/research/generated
pytest -q benchmarks/neural_ms2lda/tests
```

The packager refuses an existing output directory. Never edit a sealed raw run
or committed evidence package in place; create a new run and retain its new
reproduction ID.

## What remains

No model change is required for the current paper. Productive next evidence
would be an independently sourced dataset or a predeclared split-repetition
study. Editorial work may improve exposition, but quantitative claims must
remain generated from the sealed evidence package.

# Contextual Sparse ETM study

This package contains the scientific implementation and clean-room evaluation
for Contextual Sparse ETM, an Embedded Topic Model (ETM) adapted to short
tandem-mass-spectrometry documents.

The published ETM supplies the embedded topic--word decoder, Gaussian
variational encoder, multinomial reconstruction objective and analytic
standard-normal KL divergence. The study adds three explicit operations:

1. each topic assigns half of its probability mass to fragment words and half
   to neutral-loss words;
2. count-weighted top-2 assignments from leave-one-out token context shift the
   posterior mean through one learned scalar; and
3. 1.5-entmax maps the latent document vector to an exactly sparse topic
   mixture.

The numerical model is expressed as named tensor functions in
`contextual_sparse_etm.py`. `ContextualSparseETM` is only the small PyTorch
parameter container required for optimization and serialization. The canonical
ETM and channel-balanced ETM controls live in `etm_baselines.py` and reuse the
same decoder function where their mathematics overlaps.

## Scientific workflow

`study_protocol.py` is the single source of truth for method names, seeds,
synthetic formulations and evidence filenames. `protocol.json` contains the
data, SGNS, evaluation, Tomotopy and chemistry settings.

The execution order is declared in `reproduction_plan.py`:

- acquire the public MSnLib and annotation assets;
- create a deterministic connectivity-group train/validation/test split;
- train the vocabulary and SGNS coordinates on training spectra only;
- prepare and fit four truth-known synthetic formulations;
- fit canonical ETM, channel-balanced ETM and Contextual Sparse ETM on CUDA;
- fit Tomotopy LDA as the published non-neural comparator;
- score validation chemistry and freeze every fitted model;
- expose the test matrices only after that freeze; and
- evaluate the frozen models, audit the artifacts and build the compact
  evidence package.

No test artifact is visible to model fitting or validation. Runtime values are
reported only within a backend and are not used to compare ETM with Tomotopy.

## Maintained implementation

- `contextual_sparse_etm.py`: channel-balanced decoder, contextual evidence,
  posterior offset, Gaussian KL and 1.5-entmax equations.
- `etm_baselines.py`: canonical and channel-balanced published ETM controls.
- `topic_model_training.py`: normalized encoder input and raw-count
  reconstruction objective.
- `synthetic_msms.py`: truth-known short-spectrum generator and matching
  metrics.
- `data.py`, `spectra.py`: data preparation and leakage-controlled splitting.
- `mag.py`, `chemical.py`: leakage-filtered annotation and substructure overlap
  scoring.
- `reproduction_audit.py`: chronology, hash, probability and split-boundary
  checks.
- `tests/`: equation correspondence, data-boundary and evidence-contract tests.

The report and reviewed PDF are
`docs/research/contextual_sparse_etm_report.tex` and
`docs/research/contextual_sparse_etm_report.pdf`. The committed evidence package
is `research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/`.

## Verification

Run from the repository root in the recorded `ms2lda-neural` environment:

```bash
pytest -q benchmarks/neural_ms2lda/tests
black --check benchmarks/neural_ms2lda scripts
ruff check --config benchmarks/neural_ms2lda/ruff.toml \
  benchmarks/neural_ms2lda scripts
python -m scripts.generate_contextual_sparse_etm_report
```

The complete clean-room workflow is:

```bash
python -m scripts.run_contextual_sparse_etm_reproduction initialize --root RUN
python -m scripts.run_contextual_sparse_etm_reproduction run --root RUN
python -m scripts.package_contextual_sparse_etm_reproduction \
  --root RUN --output EVIDENCE
python -m scripts.generate_contextual_sparse_etm_report \
  --evidence-root EVIDENCE
```

# Contextual Sparse ETM paper

`contextual_sparse_etm_report.tex` is the canonical scientific report and
`contextual_sparse_etm_report.pdf` is its reviewed rendering.

The manuscript starts from the published Embedded Topic Model (ETM), explains
each domain adaptation and its rationale, derives the complete model, describes
the MSnLib preparation and leakage controls, and reports synthetic ablations,
held-out chemical evaluation and training-seed robustness. It describes the
model in its current form; superseded experimental history is outside the
paper's scope.

The implementation mirrors the mathematical notation:

- `benchmarks/neural_ms2lda/contextual_sparse_etm.py` contains the decoder,
  contextual-evidence, posterior-offset, KL and 1.5-entmax equations as named
  tensor functions;
- `benchmarks/neural_ms2lda/etm_baselines.py` contains the published ETM
  controls;
- `benchmarks/neural_ms2lda/topic_model_training.py` contains the normalized
  input and reconstruction objective; and
- `scripts/run_contextual_sparse_etm.py` performs real-data training and
  deterministic inference.

## Evidence boundary

Models are fitted on training spectra and developed on validation. Their
weights and validation outputs are frozen before the test matrices are
exposed. Test evaluation performs no fitting or model selection. The committed
evidence package is:

`research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/`

The evidence inventory and figure/table design map are in
`contextual_sparse_etm_report_sources.md`.

## Regeneration

From the repository root:

```bash
python -m scripts.generate_contextual_sparse_etm_report
pytest -q benchmarks/neural_ms2lda/tests
```

The generator validates the package seal and regenerates these complete
fragments atomically:

- `generated/contextual_sparse_etm_macros.tex`
- `generated/contextual_sparse_etm_synthetic_table.tex`
- `generated/contextual_sparse_etm_high_k_table.tex`
- `generated/contextual_sparse_etm_test_table.tex`
- `generated/contextual_sparse_etm_stability_table.tex`
- `generated/contextual_sparse_etm_diagnostics_table.tex`
- `generated/contextual_sparse_etm_hyperparameters_table.tex`
- `generated/contextual_sparse_etm_code_table.tex`

Compile with a current Tectonic or TeX Live installation, render every PDF page
to images, and inspect the layout before committing the PDF.

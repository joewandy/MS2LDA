# Contextual Sparse ETM paper

neural_ms2lda_report.tex is the canonical scientific report and
neural_ms2lda_report.pdf is its reviewed rendering.

The paper is self-contained. It starts from the published ETM, identifies each
domain-specific extension and its rationale, then reports the complete MSnLib
preparation pipeline, component ablations, held-out chemical evaluation,
training-seed robustness, system load, limitations and reproducibility.
The final layout uses a data-flow figure, an equation-matched model diagram,
chemical-filter inventory bars and a breadth--quality plot where each visual
answers a distinct scientific question.

The maintained implementation is intentionally direct:

- `benchmarks/neural_ms2lda/contextual_sparse_etm.py` exposes the decoder,
  contextual evidence, posterior offset, Gaussian KL and 1.5-entmax equations
  as named tensor functions;
- `benchmarks/neural_ms2lda/topic_model_training.py` exposes the normalized
  encoder input and raw pseudo-count reconstruction equation as plain
  functions;
- `ContextualSparseETM` is the only stateful shell and exists solely to register
  PyTorch parameters and preserve checkpoint compatibility; and
- `scripts/run_contextual_sparse_etm.py` performs training and deterministic
  inference without importing any experimental-ablation model or campaign
  runner.

## Evidence boundary

Models are fitted on the training split and developed on validation. Their
weights and validation outputs are frozen before the test matrices are exposed.
The final comparison and three-seed robustness results report that held-out test
evaluation; test evaluation performs no fitting or model selection.

The single committed evidence package is:

- research/etm_ecrtm_msnlib/local_results/20260831_contextual_sparse_etm_reproduction/

The source inventory and figure/table design map are recorded in
routing_etm_report_sources.md.

## Regeneration and verification

From the repository root:

    python -m scripts.run_contextual_sparse_etm_reproduction initialize --root RUN
    python -m scripts.run_contextual_sparse_etm_reproduction run --root RUN
    python -m scripts.package_contextual_sparse_etm_reproduction --root RUN --output EVIDENCE
    python -m scripts.generate_routing_etm_report --evidence-root EVIDENCE
    pytest -q benchmarks/neural_ms2lda/tests

The generator validates cross-file identities before writing:

- generated/routing_etm_macros.tex
- generated/routing_etm_synthetic_table.tex
- generated/routing_etm_high_k_table.tex
- generated/routing_etm_test_table.tex
- generated/routing_etm_stability_table.tex
- generated/routing_etm_diagnostics_table.tex
- generated/routing_etm_hyperparameters_table.tex
- generated/routing_etm_code_table.tex

Compile from docs/research with a current Tectonic or TeX Live installation,
then render every PDF page to images and inspect the final layout before
committing the PDF.

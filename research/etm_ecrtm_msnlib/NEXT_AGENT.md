# Continuation rules for Contextual Sparse ETM

## Start here

The model is frozen and the held-out test reproduction is complete. Read, in
order:

1. `docs/research/neural_ms2lda_report.pdf`
2. `benchmarks/neural_ms2lda/FINAL_MODEL_SELECTION.md`
3. `research/etm_ecrtm_msnlib/HANDOFF.md`
4. `local_results/20260901_contextual_sparse_etm_reproduction/README.md`
5. `local_results/20260901_contextual_sparse_etm_reproduction/acceptance.json`
6. `local_results/20260901_contextual_sparse_etm_reproduction/data_quality.json`

## Preserve these boundaries

- Do not alter the sealed raw run or compact evidence package.
- Do not treat validation results as final results; the paper's real-data table
  is the frozen held-out test comparison in `comparison.csv`.
- Do not tune or fit on test data.
- Do not replace generated paper values by manually typed numbers.
- Do not describe evaluable breadth, conditional SOS and completion NLL as one
  composite ranking; they answer different scientific questions.
- Distinguish 18 truth-matched planted motifs from 19 fitted top-1 winner topics
  in the K=128 stress experiment.
- Preserve CUDA execution for all neural variants in a new reproduction unless
  a new protocol explicitly changes the hardware question.

## Verify before editing quantitative claims

```bash
pytest -q benchmarks/neural_ms2lda/tests

python -m scripts.package_contextual_sparse_etm_reproduction \
  --root /path/to/completed-run \
  --output /path/to/new-package

python -m scripts.generate_routing_etm_report \
  --evidence-root /path/to/new-package \
  --output /path/to/new-generated-directory
```

The packager must report both `claim_checks_passed: true` and
`data_quality: pass`. A fresh generator output should match the current eight
`routing_etm_*.tex` fragments when it consumes the canonical final package.

## Allowed next work

- Editorial clarification that does not change scientific meaning.
- Independent-data evaluation with a predeclared protocol.
- Split-repetition robustness with new split seeds and untouched final evidence.
- Chemical expert review of selected motifs, clearly separated from the
  automated SOS endpoint.

Any new experiment must have its own raw root, reproduction ID and compact
package. Keep the present report and evidence as the fixed reference point.

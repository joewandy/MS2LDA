# Contextual Sparse ETM final handoff

The clean-room reproduction, evidence package and scientific report are
complete. The selected model is Contextual Sparse ETM: a published ETM
backbone with channel-balanced topic--word probabilities, one-scalar
leave-one-out contextual top-2 posterior evidence and 1.5-entmax document
mixtures.

## Final evidence

- Reproduction ID: `c8e26d27-861e-42ed-a04f-ca5a9ecfedae`
- Raw source commit: `379ab80ab6f0916e0c31a036bab229dda1d4727a`
- Raw run:
  `/home/joewandy/Work/data/MS2LDA-reproductions/20260901-379ab80-gpu-clean`
- Compact package:
  `local_results/20260901_contextual_sparse_etm_reproduction/`
- Manuscript: `docs/research/neural_ms2lda_report.tex`
- Reviewed PDF: `docs/research/neural_ms2lda_report.pdf`

The raw run completed all 54 planned stages. The compact package reports
`acceptance_all_passed: true`, `data_quality_status: pass`, and confirms that
test data were exposed only after every fitted model and required validation
output was frozen.

## Held-out test conclusion

| Model | Evaluable | Useful | Mean SOS | Completion NLL |
| --- | ---: | ---: | ---: | ---: |
| Canonical ETM | 171 | 101 | 0.631759 | **8.686003** |
| Balanced ETM | 207 | 130 | 0.644232 | 8.779686 |
| **Contextual Sparse ETM** | **572** | **343** | 0.637702 | 9.535540 |
| Tomotopy LDA | 319 | 188 | **0.652752** | 9.739090 |

The proposed model's principal gain is discovery breadth. It does not dominate
every metric, and the report retains the likelihood and conditional-SOS
trade-offs explicitly.

Three final-model training seeds preserve the result: 557--582 evaluable,
327--353 useful, 914--922 unique top-1 topics and median 3.68--3.74 effective
topics per spectrum. Every run is finite and has zero MAG exceptions.

## Reproduction guarantees

- 41,568 public source spectra and 38,888 retained spectra.
- 27,222/3,889/7,777 train/validation/test spectra.
- 38,465 connectivity groups and 28,572 split groups.
- Zero compound or group leakage.
- Training-only 21,233-word vocabulary and SGNS coordinates.
- Identical validation and test views for every comparator.
- CUDA recorded for every neural training and inference stage.
- Finite, non-negative, row-normalized paper-facing beta and theta matrices.
- Zero MAG clustering and optimization exceptions.
- SOS band counts reconcile exactly with evaluable and useful totals.

## Model boundary

The current paper needs no further architecture search. The model contains no
selectable experimental mechanism and only one additional learned scalar beyond
balanced ETM. If new scientific work begins, use a new reproduction directory,
new evidence package and predeclared question. Do not overwrite the final
package or regenerate its numbers from another run under the same path.

The next meaningful validation target is independent data or a predeclared
split-repetition study. Manuscript editing may continue, but generated
quantitative tables and macros must remain sourced from the sealed package.

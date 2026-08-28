# Neural MS2LDA research reports

`neural_ms2lda_report.tex` is the canonical report. It now combines:

- the locked M1 architecture and Tomotopy comparison;
- the validation-only ETM, pooled, balanced-ETM, and ECRTM model-selection campaign;
- the expanded collapse-diagnostic contract;
- the decision to pause architecture search and run M1 optimization-seed stability next.

The pre-model-selection detailed methods report is preserved verbatim at:

`archive/neural_ms2lda_report_pre_model_selection.tex`

Generate committed numerical fragments with:

```bash
python scripts/generate_neural_ms2lda_report.py
python scripts/generate_neural_ms2lda_model_selection.py
```

Then compile `neural_ms2lda_report.tex` and visually review every page. The old
tracked PDF was intentionally removed because it represented the pre-selection
source and would otherwise be misleading. Commit a replacement PDF only after a
local deterministic build and visual review.

The external model-selection source is:

`research/etm_ecrtm_msnlib/local_results/20260827_followup/comparison.csv`

No alternative candidate test result was used in the model-selection section.

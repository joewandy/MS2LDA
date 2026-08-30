# Neural MS2LDA research reports

`neural_ms2lda_report.tex` is the canonical report. It now combines:

- the locked M1 and Tomotopy comparison;
- the synthetic and real routing-informed sparse ETM campaign;
- the validation-only ETM, pooled, balanced-ETM, sparse-ETM, ECRTM, and NSTM
  negative evidence;
- the expanded sparsity and collapse-diagnostic contract; and
- the decision to freeze Routing ETM as the paper-facing validation baseline and
  the three-seed evidence that its discovery breadth and residual trade-offs are
  stable to initialization on the frozen split;
- the stopped weight-1 positive-NPMI experiment showing that the simplest
  coherence add-on improved its own graph statistic but not planted beta
  recovery; and
- the stopped zero-parameter top-2 token ablation showing that leave-one-out
  spectral context is necessary despite costing only one learned scalar.

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

The current checkpoint sources are:

- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/checkpoint_manifest.json`
- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/comparison.csv`
- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/metrics.json`
- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_stability/stability_summary.json`
- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_stability/checkpoint_manifest.json`
- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_npmi/synthetic_summary.csv`
- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_npmi/provenance.json`
- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_top2_token/synthetic_summary.csv`
- `research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_top2_token/provenance.json`
- `benchmarks/neural_ms2lda/results/seed42/results.json`

Verify them with `scripts/verify_routing_etm_checkpoint.py` and
`scripts/verify_routing_etm_stability.py`. No Routing ETM candidate test result
was used in the report. The stopped NPMI result is verified with
`scripts/verify_routing_etm_npmi.py`; it never reached real validation.
The zero-parameter ablation is verified with
`scripts/verify_routing_etm_top2_token.py`; it also stopped before later
synthetic stages or real validation.

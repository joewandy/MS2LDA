# Contextual Sparse ETM report sources

## Reporting contract

- Audience: technical scientific reader.
- Question: can an ETM-based model produce sparse per-spectrum mixtures while
  retaining a broad, chemically useful Mass2Motif inventory?
- Scope: truth-known synthetic experiments and one scaffold/compound-disjoint
  positive-mode MSnLib train/validation/test partition.
- Comparison basis: canonical fixed-SGNS ETM, fragment/loss-balanced ETM and
  Tomotopy LDA on the same real split; controlled ETM component variants on
  synthetic data.
- Evidence boundary: models are fitted on training data, developed on validation
  data, frozen and then evaluated once on the held-out test split.
- Primary interpretation: discovery breadth, predictive fit and conditional
  chemical quality are separate axes and are reported together.
- Chemical comparison rule: every held-out spectrum is associated with its
  single dominant full-spectrum topic for every model.

## Scientific-report structure map

| Technical-report role | Paper section |
| --- | --- |
| Title and technical summary | Title and Abstract |
| Key findings with visual evidence | Results |
| Scope, data and metric definitions | Materials and methods: Dataset construction and leakage control; Evaluation measures |
| Model specification and experimental design | Materials and methods: Contextual Sparse ETM; Experimental design and comparators |
| Robustness and uncertainty | Results: Contextual Sparse ETM expands the chemically assessable motif inventory; Limitations |
| Recommended next evidence | Discussion: Breadth--quality trade-offs and generalization priorities |
| Further questions | Limitations |

## Reader-facing terminology contract

- Expand uncommon abbreviations at first use in the abstract or main text;
  repeat the expansion in a figure or table caption when the item should stand
  alone.
- Figure 3 defines ETM, Balanced, Contextual and LDA, and states the shared
  dominant-topic association rule, in the caption rather than relying on the
  experimental-design section.
- Quantitative table captions define NLL, SOS, effective-topic counts, support
  and winner counts before the reader interprets their columns.
- Describe implementation terms by function on first use: MGF as the spectrum
  file format, CSR as sparse matrix storage, FAISS as vector-similarity search,
  MACCS as binary molecular-feature fingerprints, and CUDA allocation versus
  reservation as distinct GPU-memory measurements.

## Source inventory

| Evidence | Source |
| --- | --- |
| Executable model equations | benchmarks/neural_ms2lda/contextual_sparse_etm.py |
| Normalized encoder input and reconstruction equation | benchmarks/neural_ms2lda/topic_model_training.py |
| Deterministic real-data inference | scripts/run_contextual_sparse_etm.py |
| Dominant-topic chemical association and SOS calculation | benchmarks/neural_ms2lda/chemical.py |
| Equation-level correspondence and serialized-state parity | benchmarks/neural_ms2lda/tests/test_contextual_sparse_etm.py |
| Data parsing, split, leakage and vocabulary counts | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/preparation_summary.json |
| Preprocessing, SGNS, Tomotopy and MAG settings | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/protocol.json |
| Proposed-model configuration and final test metrics | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/config.json and metrics.json |
| Synthetic K=36 component isolation | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/synthetic_summary.csv |
| Synthetic K=128 stress | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/high_k_stress.csv |
| Final real-model comparison | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/comparison.csv |
| Development-split comparison | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/validation_comparison.csv |
| Three-seed test robustness | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/stability_summary.json |
| Integrity and provenance | fresh_evidence_manifest.json, checkpoint_manifest.json, reproduction_manifest.json and stage_records/ in the same package |

## Figure and table map

| Item | Question | Form | Supported claim | Palette and QA |
| --- | --- | --- | --- | --- |
| Figure 1 | How do public assets become model matrices? | Process flow | Acquisition, validation, splitting and sparse-matrix hand-off are explicit | Neutral grey; inspect arrows and box fit |
| Figure 2 | Where does context modify ETM? | Computation flow | Contextual evidence adjusts the posterior while the ETM generator remains explicit | Blue for posterior, violet for shared geometry, green for outputs; inspect crossings |
| Figure 3 | At which chemical filter does inventory breadth differ? | Three-panel categorical bar chart | The proposed model's advantage is evaluable/useful inventory, not merely optimized count | Proposed blue, Tomotopy orange, ETM baselines neutral; zero baselines and direct labels |
| Figure 4 | How does evaluable breadth relate to conditional chemical quality? | Labelled scatter plot | Contextual Sparse ETM expands breadth while Tomotopy retains higher conditional mean SOS | Shared axes, direct labels and no composite score |
| Table 1 | Which parts are inherited from ETM and which are added? | Lineage table | The generator and variational backbone remain ETM; the adaptations have explicit purposes | Inspect provenance and parameter claims |
| Tables 2--3 | Which posterior additions are necessary? | Exact ablation tables | Contextual evidence prevents the component loss caused by sparsity alone | No conditional color; inspect width |
| Table 4 | How do held-out test metrics compare? | Exact comparison table | Breadth, SOS and NLL point in different directions | Bold per-column extrema; inspect that caption explains this |
| Table 5 | Does initialization change the result? | Exact robustness table | Central sparsity and breadth persist across three seeds | Resized full-width table; inspect readability |
| Table 6 | Is local sparsity caused by global collapse? | Diagnostic table | Sparse spectra coexist with broad topic use | Exact values; inspect long row labels |
| Tables 7--9 | Can settings, code correspondence and metric meanings be audited? | Appendix tables | The reported model and evidence can be reproduced and interpreted without hidden conventions | Inspect page fit and monospaced paths |

The source inventory is limited to the sealed acquisition, synthetic,
validation and frozen-model test evidence listed above.

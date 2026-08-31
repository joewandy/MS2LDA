# Contextual Sparse ETM report sources

## Reporting contract

- Audience: technical scientific reader.
- Question: can an ETM-based model produce sparse per-spectrum mixtures while
  retaining a broad, chemically useful Mass2Motif inventory?
- Scope: truth-known synthetic experiments and one scaffold/compound-disjoint
  positive-mode MSnLib training/validation split.
- Comparison basis: canonical fixed-SGNS ETM, fragment/loss-balanced ETM and
  Tomotopy LDA on the same real split; controlled ETM component variants on
  synthetic data.
- Evidence boundary: the proposed model's test partition is untouched.
- Primary interpretation: discovery breadth, predictive fit and conditional
  chemical quality are separate axes and are reported together.

## Scientific-report structure map

| Technical-report role | Paper section |
| --- | --- |
| Title and technical summary | Title and Abstract |
| Key findings with visual evidence | Results |
| Scope, data and metric definitions | Materials and methods: Dataset construction and leakage control; Evaluation measures |
| Model specification and experimental design | Materials and methods: Contextual Sparse ETM; Experimental design and comparators |
| Robustness and uncertainty | Results: Real-data motif discovery and initialization stability; Limitations |
| Recommended next evidence | Discussion: Breadth--quality trade-offs and validation priorities |
| Further questions | Limitations and validation priorities |

## Reader-facing terminology contract

- Expand uncommon abbreviations at first use in the abstract or main text;
  repeat the expansion in a figure or table caption when the item should stand
  alone.
- Figure 3 defines ETM, Balanced, Contextual and LDA in the caption rather than
  relying on the experimental-design section.
- Quantitative table captions define NLL, SOS, effective-topic counts, support
  and winner counts before the reader interprets their columns.
- Describe implementation terms by function on first use: MGF as the spectrum
  file format, CSR as sparse matrix storage, FAISS as vector-similarity search,
  MACCS as binary molecular-feature fingerprints, and CUDA allocation versus
  reservation as distinct GPU-memory measurements.

## Source inventory

| Evidence | Source |
| --- | --- |
| Data parsing, split, leakage and vocabulary counts | research/etm_ecrtm_msnlib/local_results/20260827_seed42_validation/preparation_summary.json |
| Preprocessing, SGNS, Tomotopy and MAG settings | benchmarks/neural_ms2lda/protocol.json |
| Proposed-model configuration and validation metrics | research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/config.json and metrics.json |
| Synthetic K=36 component isolation | research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/synthetic_summary.csv |
| Synthetic K=128 stress | research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/high_k_stress.csv |
| Real ETM baseline comparison | research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/comparison.csv |
| Tomotopy validation comparator | benchmarks/neural_ms2lda/results/seed42/results.json |
| Three-seed real robustness | research/etm_ecrtm_msnlib/local_results/20260830_routing_etm_stability/stability_summary.json |
| Integrity and provenance | Both proposed-model checkpoint manifests and verification scripts |

## Figure and table map

| Item | Question | Form | Supported claim | Palette and QA |
| --- | --- | --- | --- | --- |
| Figure 1 | How do public assets become model matrices? | Process flow | Acquisition, validation, splitting and sparse-matrix hand-off are explicit | Neutral grey; inspect arrows and box fit |
| Figure 2 | Where does context modify ETM? | Computation flow | Contextual evidence adjusts the posterior while the ETM generator remains explicit | Blue for posterior, violet for shared geometry, green for outputs; inspect crossings |
| Figure 3 | At which chemical filter does inventory breadth differ? | Three-panel categorical bar chart | The proposed model's advantage is evaluable/useful inventory, not merely optimized count | Proposed blue, Tomotopy orange, ETM baselines neutral; zero baselines and direct labels |
| Figure 4 | How does evaluable breadth relate to conditional chemical quality? | Labelled scatter plot | Contextual Sparse ETM expands breadth while Tomotopy retains the highest conditional mean SOS | Shared axes, direct labels and no composite score |
| Table 1 | Which parts are inherited from ETM and which are added? | Lineage table | The generator and variational backbone remain ETM; the adaptations have explicit purposes | Inspect provenance and parameter claims |
| Tables 2--3 | Which posterior additions are necessary? | Exact ablation tables | Contextual evidence prevents the component loss caused by sparsity alone | No conditional colour; inspect width |
| Table 4 | How do real validation metrics compare? | Exact comparison table | Breadth, SOS and NLL point in different directions | Bold per-column extrema; inspect that caption explains this |
| Table 5 | Does initialization change the result? | Exact robustness table | Central sparsity and breadth persist across three seeds | Resized full-width table; inspect readability |
| Table 6 | Is local sparsity caused by global collapse? | Diagnostic table | Sparse spectra coexist with broad topic use | Exact values; inspect long row labels |
| Tables 7--9 | Can settings, code correspondence and metric meanings be audited? | Appendix tables | The reported model and evidence can be reproduced and interpreted without hidden conventions | Inspect page fit and monospaced paths |

The source inventory is limited to the admissible training, synthetic and
validation evidence listed above.

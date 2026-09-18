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
  data, frozen and then evaluated once on the held-out test split in the historical
  sealed study. The subsequent simplification review is separately labelled
  validation development after those test results were known; its new candidates
  never load test matrices.
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
| Staged independent review protocol and publication recommendation | research/minimal_neural_etm/review_20260907/README.md |
| Initial review: all 24 synthetic and three real fits | research/minimal_neural_etm/review_20260907/evidence/runs.csv and manifest.json |
| Earlier 53 records, including normalization sensitivity | research/minimal_neural_etm/review_20260907/evidence/prior_round/runs.csv and manifest.json |
| Expanded published families: 30 synthetic and two real fits | research/minimal_neural_etm/review_20260907/evidence/published/runs.csv and manifest.json |
| Numerical summaries, descriptive SDs, all joint decision gates | review_summary.json in the initial and published evidence directories |
| Published-family probability/gradient checks | benchmarks/neural_ms2lda/tests/test_published_models.py |
| Complete-inventory and meaning checks | benchmarks/neural_ms2lda/tests/test_minimal_etm_review.py and test_published_review.py |
| Atomic reductions, advancement tree and combination protocol | research/minimal_neural_etm/review_20260907/within_model_protocol.md |
| Within-model fits, failed deletions, seeds and joint gates | research/minimal_neural_etm/review_20260907/evidence/within_model/runs.csv, manifest.json and review_summary.json |
| Explicit numerical failure, source hashes, and separate execution interruption | review evidence/within_model/failed_runs.json and interrupted_attempts.json |
| Exact log1p, channel-conditional likelihood and Gaussian-mean identities | scripts/audit_contextual_reduction_identities.py and evidence/within_model/mathematical_audit.json in the review |
| Reduced models and independent gradient/initialization tests | benchmarks/neural_ms2lda/contextual_reductions.py and tests/test_contextual_reductions.py |
| Staged-inventory completeness and numerical threshold equality | scripts/generate_contextual_reduction_review.py and benchmarks/neural_ms2lda/tests/test_contextual_reduction_review.py |
| SOS-cutoff sensitivity and paired effect sizes (not a new acceptance rule) | scripts/summarize_minimal_etm.py, scripts/generate_contextual_reduction_review.py and the within-model CSV/summary |
| Independent CPU reload of GPU-trained validation predictions | review evidence/within_model/checkpoint_reload_audit.json |
| All three recommended full-document checkpoints: frozen inputs, probability and CPU/GPU checks | review evidence/within_model/recommended_checkpoint_reload_audit.json |
| Replaceable MLP-free posterior encoder and raw-input contract test | benchmarks/neural_ms2lda/attention_etm.py and tests/test_attention_etm.py |
| Frozen MLP-free state import and full-validation numerical parity | scripts/audit_attention_etm_interface.py and review evidence/within_model/attention_interface_audit_seed42.json |

## Figure and table map

| Item | Question | Form | Supported claim | Palette and QA |
| --- | --- | --- | --- | --- |
| Figure 1 | How do public assets become model matrices? | Process flow | Acquisition, validation, splitting and sparse-matrix hand-off are explicit | Neutral grey; inspect arrows and box fit |
| Figure 2 | Where does context modify ETM? | Computation flow | Contextual evidence adjusts the posterior while the ETM generator remains explicit | Blue for posterior, violet for shared geometry, green for outputs; inspect crossings |
| Figure 3 | At which chemical filter does inventory breadth differ? | Three-panel categorical bar chart | The proposed model's advantage is evaluable/useful inventory, not merely optimized count | Proposed blue, Tomotopy orange, ETM baselines neutral; zero baselines and direct labels |
| Figure 4 | How does evaluable breadth relate to conditional chemical quality? | Labelled scatter plot | Contextual Sparse ETM expands breadth while Tomotopy retains higher conditional mean SOS | Shared axes, direct labels and no composite score |
| Table 1 | Which parts are inherited from ETM and which are added? | Lineage table | ETM supplies the embedding decoder and Gaussian latent variational backbone; channel normalization and entmax change the generative distribution | Inspect provenance and parameter claims |
| Tables 2--3 | What does the original bundled posterior ablation show? | Exact ablation tables | Contextual evidence improves recovery relative to the original sparse control; individual necessity is not isolated | No conditional color; inspect width |
| Table 4 | How do held-out test metrics compare? | Exact comparison table | Breadth, SOS and NLL point in different directions | Bold per-column extrema; inspect that caption explains this |
| Table 5 | Does initialization change the result? | Exact robustness table | Central sparsity and breadth persist across three seeds | Resized full-width table; inspect readability |
| Table 6 | Is local sparsity caused by global collapse? | Diagnostic table | Sparse spectra coexist with broad topic use | Exact values; inspect long row labels |
| Initial and published-family review tables | Does a simpler neural family retain real chemical breadth and compactness? | Validation comparison tables | No phase-1/2 candidate passes all replacement criteria; published-family likelihoods are scored correctly | Neutral exact values, explicit recipe/prototype differences |
| Supporting simplification tables | Which simplifications succeed synthetically but fail to transfer? | Supporting review tables | Sparse mixtures, recovery, duplication and chemical usefulness are distinct | All candidates retained, no selective best metric |
| Remaining appendix tables | Can settings, code correspondence and metric meanings be audited? | Settings/code/metric tables | The reported model and evidence can be reproduced and interpreted without hidden conventions | Inspect page fit and monospaced paths |
| Within-model reduction tables | Which individual choices can be simplified, and do their combinations transfer? | Exact staged ablation tables | Synthetic passes and real chemical breadth must be distinguished; no absolute-minimality assertion follows from a finite screen | Include every scheduled group and seed, no favorable-metric substitution |
| Within-model repeat table | Which simplification is supported across matched seeds? | Mean and sample-SD table | Whole-spectrum context preserves useful breadth with slightly better NLL while keeping ETM's MLP; attention-only and shallower MLPs have different trade-offs | Neutral values; three seeds are not external confidence bounds |
| Within-model SOS table | Does the interpretation depend only on SOS 0.6? | Inclusive threshold-count table | Whole-spectrum context has more motifs at SOS 0.7 and 0.8 in all three pairs; thresholds are correlated summaries | Show all five fixed cutoffs, failed evaluation explicitly missing, no favorable cutoff selection |

Historical test numbers remain sourced only from the sealed evidence. Review
numbers are sourced only from their separate validation inventories; no review
candidate inherits the historical model's test score. The report generators
are separately executable with `python -S` and record numerical-source hashes.

The 8 September closeout comprises 87 synthetic and 20 successful real phase-3
fits plus one explicit real numerical failure. The 166 successful new fits
across all three phases are separate from the 53 earlier audited records.
The final recommendation is full-document context with the conventional
two-layer MLP, not the attention-only boundary model. Width 100 and fixed
context weight remain plausible further simplifications with only one real
seed each. No claim of absolute minimality or automatic decision by the old
reference flags is supported. Exact comparison tables are used for this audit;
there is no new composite-score chart or new runtime claim from concurrent jobs.

Additional primary-source precedents inspected during the within-model review:

- DreaMS [official repository](https://github.com/pluskal-lab/DreaMS) and
  [embedding tutorial](https://dreams-docs.readthedocs.io/en/latest/tutorials/compute_embeddings.html),
  linked to Bushuiev et al. [Nature Biotechnology 2025](https://doi.org/10.1038/s41587-025-02663-3):
  spectrum-level self-supervised representations and 1024-dimensional output
  motivate a future encoder adapter. This is a design direction, not an
  installed dependency, evaluated baseline or empirical performance claim.
  The inference-encoder swap and fully tokenizer-free likelihood are explicitly
  different scopes; pretrained-corpus overlap needs its own audit. The journal
  page was not directly retrievable in this session; the factual interface
  claims are supported by the authors' repository and documentation.
  The [official API source](https://github.com/pluskal-lab/DreaMS/blob/main/dreams/api.py)
  additionally shows that the convenience embeddings call selects a
  `ContrastiveHead`, and applies spectrum preprocessing/peak selection. A
  future experiment must distinguish this from the self-supervised backbone
  and audit both pretraining and fine-tuning provenance.

- Mimno et al., EMNLP 2009, [Polylingual Topic Models](https://aclanthology.org/D09-1092.pdf):
  shared mixture and view-specific emissions provide a published structural
  precedent for the two-channel conditional interpretation. The ETM embedding
  constraints, sparse Gaussian-derived prior and neural inference remain
  distinct; this is lineage, not a claimed PLTM reproduction.

- Lin, Hu and Guo, WSDM 2019, [Gaussian-sparsemax topic models](https://arxiv.org/pdf/1810.09079):
  sparse simplex mappings have direct topic-model precedent. Their relaxed-Wasserstein
  objective is not the Gaussian latent-space ELBO used here. Not newly fitted.
- Panwar et al., ACL 2021, [TAN-NTM](https://aclanthology.org/2021.acl-long.299.pdf):
  topic-aware attention can inform variational recognition. Its recurrent
  sequence encoder is distinct from this order-invariant spectral encoder.
  Not newly fitted.
- Chen et al., ECML PKDD 2023, [SpareNTM](https://ecmlpkdd-storage.s3.eu-central-1.amazonaws.com/preprints/2023/research/lncs14172155.pdf):
  an explicit Bernoulli selector and non-mean-field Dirichlet construction offer
  a published sparse alternative but add latent machinery. Not newly fitted;
  no real-data performance is inferred from the literature.

The mathematical audit clarifies the Gaussian-to-entmax induced prior, the
latent-space ELBO without a Jacobian, plug-in versus integrated prediction, and
raw versus unit-normalized topic vectors in the computational diagram. The
published-family audit additionally distinguishes actual PoE word probabilities
from additive emissions and checks that validation cannot update BatchNorm.

# Discovering Molecular Substructures from Mass Spectra with Neural Topic Models: sources

## Current document map - 18 September 2026

The canonical deliverable is now a pair: the [main paper](contextual_sparse_etm_report.pdf)
and [Supplementary Information](contextual_sparse_etm_supplement.pdf). The main
paper retains five numbered equations and all three explanatory figures; full
technical formulations and protocols have explicit point-of-use supplementary
references. Both documents share [one bibliography source](contextual_sparse_etm_references.tex).
Keep the PDFs together for the numbered cross-document links to work.

| Material | Current location |
| --- | --- |
| Base ETM and three enhancements | Main Methods 3.2-3.6, Equations 1-5, Figures 2-3 |
| Spectral representation and worked tokenization example | Main Methods 3.1 and Figure 1; full details and notation in S1 |
| Complete model equations and numerical conventions | Supplementary S2, Equations S1-S11 |
| LDA/ETM component correspondence and comparator protocols | S3, Table S2; summarized in main Methods 3.7 |
| Full evaluation definitions | S4, Equations S12-S14; explained in main Methods 3.8 |
| Chemical nulls, feature enrichment and matching | S5, Equations S15-S16; summarized in main Methods 3.9 |
| Directed recovery and exact aggregation | S6, Equation S17; summarized in main Methods 3.10 |
| Chemical-agreement, multiplicity and channel-sensitivity diagnostics | S7, Figure S1; referenced in main Results 4.3 |
| Within-fit redundancy and former reproducibility appendix | S7.1 and S8 |
| Primary synthetic and real-data comparisons | Main Tables 1-2 |
| Cross-model, chemical-assessment and reference-example tables | Main Tables 3-5 |
| Recovery, actual motif counterparts, specificity and repeatability figures | Main Figures 4-7 |

The [independent editorial review](editorial_review_20260918.md) records the
allocation, preservation audit and resolved clarity findings. Dated revision
records below retain their historical page, equation, figure and table numbers;
this current map supersedes those locations for the two-document edition.
The subsequent [final scientific review](final_scientific_review_20260918.md)
and [results audit](final_results_audit_20260918.md) record the comprehensive
editorial, methodological, implementation and numerical checks.

## Reporting contract

- Audience: technical scientific reader; existing LaTeX/PDF is the explicitly
  requested artifact, so no substitute HTML, dashboard or hosted report.
- Question: how does the chosen ETM-derived model work, why are its enhancements
  sensible, and how does it differ from and compare with Tomotopy LDA?
- Primary comparator: Tomotopy LDA. Plain ETM is a secondary base-model
  check only. Other architectures and component variants stay in lab records.
- Selected model: reduced_document_context, retaining the two-hidden-layer width-800
  MLP, channel-balanced emissions, learned scalar context strength, top-2
  cosine matching, centered log evidence and 1.5-entmax mixtures.
- Evidence: six synthetic fits (K=36 and 128; seeds 11, 23, 37) and three real
  validation fits (K=1000; run seeds 11, 23, 42, actual training seeds
  7012, 7024, 7043). All fitted parameters use training spectra.
- Validation has 3,889 full-spectrum chemical associations per model. Completion
  has 3,888 eligible spectra and 1,277,983 withheld in-vocabulary pseudo-tokens;
  41,369 withheld tokens are OOV (3.14%) and excluded from NLL.
- Main interpretation: distinguish useful-motif breadth, conditional mean SOS,
  prediction and mixture concentration. Do not describe this as uniform
  superiority or claim an unmeasured runtime advantage over LDA.
- SDs are sample SDs, not standard errors or external confidence intervals.
  Real repetitions vary training initialization on fixed data/embeddings;
  synthetic repetitions vary both the simulated data and initialization.
  The retained plain-ETM fit used WSL2 and its new fits native Linux, with
  matching scientific package versions but a different Python build; that SD
  describes fixed-split run-to-run variation, not only seed variation.
  All three model rows have three fits each. Seed identifiers appear only in
  Supplementary Section S8 and machine-readable evidence.
- The plain ETM reference uses neural batch size 256 versus selected-model 200;
  its limited comparison does not isolate each enhancement's causal effect.
- No current-model test evidence exists. The previously used reserved split
  is not newly unseen data; independent post-selection evaluation is required.

## Structure and scope decisions

The user's explicit instruction preserves existing scientific section roles
and orders Methods before Results. Technical-report skill roles map as follows:

| Required role | Existing report location |
| --- | --- |
| Title and technical summary | Title and Abstract |
| Scope, data and definitions | Materials and methods: representation and evaluation |
| Published foundation and chosen specification | Main methods: ETM base, three enhancements and conceptual objective; complete mathematical specification in Supplementary S2 |
| Experimental design | Materials and methods: Tomotopy primary, plain ETM secondary, synthetic recovery |
| Findings and evidence | Results: synthetic recovery, then MSnLib with Tomotopy primary and plain ETM secondary; aggregate stability |
| Uncertainty and robustness | Repetition definitions, mean ± SD and fit counts; Discussion 5.3 on evidence limitations |
| Next steps | Discussion 5.4: independent comparison with Tomotopy, chemical assessment and future encoders; separate Conclusion |
| Further questions | Discussion 5.1–5.4: interpretation of inventory gains, component evidence, selection bias and future validation |
| Reproduction details | Short availability statement after Conclusion; Supplementary S8; this supporting inventory |

The experiment catalogue, old test results, extended alternatives discussion,
full ablation tables, and long audit appendices were removed from the main report
at the user's request. They are preserved unchanged in the archived
pre-refocus report and research notes, not silently discarded. No claimed
ablation effect from the historical architecture is relabelled as an effect
for the chosen whole-spectrum model.
The approved 8 September revision expands background, related literature,
data transformations, notation, corresponding LDA/ETM components, enhancement
examples, metric definitions and interpretation. It preserves the chosen
model and conventional scientific section roles. The subsequent sharing pass
combined Discussion, Limitations and Conclusion into one ending section and
added five baseline fits without changing selected-model evidence. The user's
following editorial request restores Conclusion as Section 6 and rewrites
Discussion as four substantive numbered subsections, without bold paragraph
headings. That structural revision left the other manuscript sections and
numerical evidence unchanged. A subsequent whole-manuscript language review
improves phrasing and transitions while preserving this structure, the equations
and the results. A short appendix explains the limited scope of existing
component evidence without reintroducing the full variant catalogue.

## Numerical evidence inventory

Paths are relative to the repository root.

| Evidence | Source and role |
| --- | --- |
| Selected-model run values | research/minimal_neural_etm/review_20260907/evidence/within_model/runs.csv; filter exactly reduced_document_context and require its nine fits |
| Complete experimental inventory and provenance | Same directory: manifest.json, failed_runs.json, interrupted_attempts.json; full staged audit remains in the generator |
| Current-model means and sample SDs | Same directory: review_summary.json, current_model_report block |
| Current Tomotopy and plain ETM baselines | research/baseline_repeats_20260908/evidence/runs.csv and manifest.json; exactly three fresh Tomotopy fits, two new plain-ETM fits and the retained original plain-ETM fit |
| Historical single-fit reference | research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/validation_comparison.csv; preserve as history, not an additional current comparison fit |
| Split, vocabulary, filtering and overlap counts | Same frozen directory: preparation_summary.json |
| Preprocessing, SGNS, LDA and MAG settings | Same frozen directory: protocol.json |
| Chosen checkpoints, exact support, completion totals, duplication and reload parity | research/minimal_neural_etm/review_20260907/evidence/within_model/recommended_checkpoint_reload_audit.json and the three chosen real run result files |
| Raw current real runs | output/benchmarks/contextual_reduction_20260907/real/reduced_document_context_seed11, seed23, and seed42_retry1 |
| Historical 29-page full audit | research/minimal_neural_etm/review_20260907/model_form_review_report_20260908.tex and .pdf |
| Historical audit source map | research/minimal_neural_etm/review_20260907/model_form_review_report_sources_20260908.md |
| Full selection reasoning | research/minimal_neural_etm/review_20260907/README.md and within_model_protocol.md |
| Scoped predecessor ablation interpretation | Same review README, Phases 1/2 decision (394 versus 322 useful motifs on balance removal), morning recommendation and within_model evidence; these are not relabelled final-model ablations |
| Simulator generation and truth construction | benchmarks/neural_ms2lda/synthetic_msms.py: _draw_document, _true_theta, _true_beta and matched_truth_metrics |

The original model comparison uses generated/current_contextual_etm_macros.tex;
chemical assessment uses generated/chemical_assessment_macros.tex, and directed
Tomotopy correspondence uses generated/cross_model_overlap_macros.tex.
None imports historical test macros. The current
comparison is generated in the order Tomotopy, plain ETM, selected model,
with fit counts 3, 3 and 3. The plain ETM reference remains the secondary
enhancement baseline. Every reference SD is computed from actual independent
training fits on the fixed partition. Full-validation Tomotopy mixture arrays
supply the previously unavailable effective-topic diagnostic. The generator still emits
per-run stability for audit, but the manuscript does not import that fragment.
All other old fragments remain available to reproduce the archived audit.

## Equation and implementation correspondence

| Report operation | Implementation |
| --- | --- |
| Published ETM controls | benchmarks/neural_ms2lda/etm_baselines.py |
| Chosen full-spectrum mean, contextual words and top-2 pooling | benchmarks/neural_ms2lda/contextual_reductions.py: pooled_evidence(context=document), selected by ReducedContextualETM with reduced_document_context |
| Balanced emission matrix | benchmarks/neural_ms2lda/contextual_sparse_etm.py: channel_balanced_topic_word_distribution |
| Centered logarithmic offset | Same module: centered_log_evidence_offset; equivalent centered log1p(Kr) verified independently |
| Gaussian KL and reparameterization | Same module: diagonal_gaussian_kl and reparameterized_gaussian |
| Sparse simplex mapping | Same module: entmax15_document_mixture |
| Actual corrected posterior | benchmarks/neural_ms2lda/contextual_reductions.py: ReducedContextualETM.posterior |
| Normalized input and raw-count reconstruction | benchmarks/neural_ms2lda/topic_model_training.py |
| Training and export | scripts/run_minimal_etm.py |
| Deterministic full/observed inference | benchmarks/neural_ms2lda/model_inference.py: infer_document_topics |
| Synthetic alignment and recovery | benchmarks/neural_ms2lda/synthetic_msms.py: matched_truth_metrics |
| MAG, dominant-topic association, fingerprint consensus and SOS | benchmarks/neural_ms2lda/chemical.py |
| Current-model table and macro generation | scripts/generate_contextual_reduction_review.py: current_model_report |
| Complete baseline inventory, source integrity, aggregation and directional emphasis | benchmarks/neural_ms2lda/report_comparison.py; tests/test_report_comparison.py in the research package |
| Independent whole-spectrum equation and gradients | benchmarks/neural_ms2lda/tests/test_current_model_equations.py and test_contextual_reductions.py |
| Report identity, seed, recipe, comparator and validation-only guardrails | benchmarks/neural_ms2lda/tests/test_contextual_reduction_review.py |

The shared contextual_sparse_etm.py preserves the original leave-one-out class.
The main report explicitly selects its whole-spectrum subclass rather than
claiming the old class became the selected model. The editorial pass did not
change model behavior. The subsequent code audit corrected a uniform-evidence
gradient boundary in the shared offset function; saved evidence is unchanged.
That distinction is documented in the code-quality record below.
The follow-up [equation audit](equation_code_audit_20260908.md) independently
checks the full selected forward calculation and derivatives, and maps every
displayed equation to code. It clarifies the 1e-12 probability floor, the
zero-observed completion row, synthetic versus real count construction, and
MAG's actual Euclidean similarity-profile clustering. None of these clarifications
changes a saved valid-input result. Matching now rejects undefined input shapes
and fewer fitted topics than planted motifs instead of silently scoring a subset.

## Figure and table contracts

Notation-only changes: E replaces the overloaded embedding-dimension L;
fragment/loss vocabularies are calligraphic V_F/V_L; lambda_ctx replaces the
scalar c without changing the code's context_scale; pi denotes local attention
instead of overloading q for both attention and the Gaussian posterior.
The LDA concentration vector is gamma, distinct from ETM topic vectors alpha.
Precursor-minus-fragment notation explicitly uses the benchmark's m/z
convention, not an inferred charge-resolved neutral composition.

| Item | Question and form | Evidence and interpretation | QA |
| --- | --- | --- | --- |
| Figure 1 | How does a physical peak become model input? Worked spectrum, count table and parallel data partitions | Illustrative 100/60 m/z peak, loss 40, intensity 0.40 gives 40 copies per channel; actual unpadded tokens frag@60.0/loss@40.0 | Two arrows only; precursor marker is not an intensity peak; encoder normalization versus count likelihood explicit |
| Figure 2 | What does each enhancement do? Three-panel schematic using composition bars and contextual flow | (a) illustrative channel mass 0.85/0.15 versus fixed 0.50/0.50; (b) same word in two hypothetical contexts; (c) identical scores under softmax and entmax | Static native TikZ; two-root blue/gold palette plus grey; segment order and values provide non-color reading; no empirical claims from schematic values |
| Figure 3 | Where do the enhancements enter ETM? Additive posterior equation and worked theta × beta = p example | Scores (1,0,-1) yield weights approximately (0.831,0.169,0); every illustrative beta row has half mass in each channel | Numerical simplex/channel checks; ≥9 pt labels; no crossing wires; shared raw versus unit topic geometry stated |
| Table 1 | What does each symbol mean? Notation lookup | Index domains, dimensions, vectors, scalar and probability normalization | Core notation defined before equations; auxiliary symbols defined locally |
| Table 2 | How do corresponding components differ? LDA/ETM/enhanced-ETM mapping | Five aligned components, with shared additive-mixture foundation | Ragged-right text for readable narrow columns; no claim LDA ignores context |
| Table 3 | Can the selected model recover planted structure? Two-K synthetic table | Six selected-model fits; means ± sample SD; 18 planted motifs, cosine threshold 0.50 | Definitions in Methods; matched mixture does not score unmatched topic mass |
| Table 4 | How does the model compare with Tomotopy and its ETM base? Three-row multi-metric table | Same validation population; fit counts 3/3/3; mean ± sample SD for every metric | Tomotopy first and primary; plain ETM secondary; best directional means bold, exact full-precision ties equal; no ranking of effective topic count |

Tables are preferable to additional quantitative charts here: the main
comparison has three models and mixed metric units. Synthetic recovery needs exact multiple metrics at
two K values. Aggregate stability is stated in prose, with per-run records
retained outside the manuscript. A bar-chart repetition of useful
counts would add little and distract from the primary multi-metric comparison.
No composite score, rank-based color, truncated count axis or fabricated
reference uncertainty is used. Table emphasis is descriptive, not inferential.
All three figures and every table have adjacent
explanatory text.

The enhancement figure is deliberately schematic: additional observations
would turn an explanatory example into an unsupported experiment. It uses
native vector primitives within the existing LaTeX/PDF rather than a raster,
HTML or independently hosted surface. Final QA inspects the full A4 pages,
including labels, consistent unit-mass bars and the absence of clipped text.

## Literature and terminology

Primary literature is cited in the manuscript for ETM, LDA/MS2LDA,
MS2LDA 2.0/MAG/SOS, entmax, training embeddings, Spec2Vec, MSnLib, scaffold
grouping and fingerprints. The plain ETM backbone and 1.5-entmax paper were
checked against the authors' published papers during this review.
Tomotopy's official repository identifies its Gibbs-sampling implementation;
the release link uses the valid v0.13.0 tag (the old unprefixed link returned
404). The unchanged archival source map retains all historical citations.
The expanded Related Work also cites MotifDB (Rogers et al., 2019), TAN-NTM
(Panwar et al., 2021), DreaMS (Bushuiev et al., online 2025 / issue 2026), and
the official Pyro ProdLDA tutorial. Primary publisher/ACL/arXiv records were
consulted. The ETM paper's background-model-inference progression informed
explanation order; MS2LDA 2.0's motivation and annotated examples informed
context and evaluation explanation. No figures or passages were copied.
MS2LDA 2.0 is credited for its existing speed and annotation improvements;
the report does not portray those older bottlenecks as unsolved by version 2.0.

MGF, MLP, MAG and SOS are expanded or defined at first use. MACCS is described
as a binary structural-key fingerprint encoding predefined molecular features.
Context attention weights are not called a chemical posterior.
The report states the change from the original MS2LDA 2.0 weight threshold
to dominant-topic evaluation, applied consistently across models.

## Historical editorial validation record (before title/code-quality revision)

No new training, test evaluation, raw-data download or production integration
is performed in this editorial pass. The full selection history and original
report have recoverable copies.

Completed 8 September 2026:

- The revised PDF has 14 pages, three figures and four tables. Every page was
  rendered and visually inspected; the final wording pass changed only pages
  2, 7 and 14, which were re-inspected. All other page PNGs were byte-identical.
- Final LaTeX compilation has no undefined references/citations, overfull or
  underfull boxes, or other warnings. The source is compiled by absolute path
  into a dedicated build directory, separate from source backups.
- All 206 scientific tests pass. The report-contract test now enforces the
  renamed context scalar and attention notation, approved figure inputs,
  aggregate-only main results, metric-before-results ordering and Conclusion
  before availability/appendix. Baseline tests enforce both references in the
  table, explicit fit counts, correct ordering and distinct baseline values.
- The exact 67-file neural-workflow Black and research-config Ruff checks pass.
  The root-wide Ruff configuration is not the configuration used by this
  research workflow; its unrelated pre-existing warnings were not auto-fixed.
- Independent standard-library aggregation of the nine selected runs
  reproduces every real and synthetic mean/sample SD in the report summaries.
  The three-topic softmax/entmax illustration was checked numerically against
  the implemented mapping, including its exact zero.
- Regeneration is byte-identical for all current-model LaTeX fragments and
  the review summary after updating generator-source provenance. No raw
  result, split, checkpoint or frozen comparison was modified.
- Pre-title PDF SHA-256:
  `0e7c57e45054796817d212eac3454cb629ab56b9ed805ff350278778c9594349`.
- The unchanged 29-page historical PDF retains SHA-256
  `089b63c270ffd2cd576f63e0345105367d47a3329e1f687b0e94d017c067169e`.

Assessment: share with the stated scientific caveats, not an independent
publication-validation claim. Single-fit baselines, reused selection data,
missing Tomotopy sparsity and annotation-proxy limitations remain visible.

## Earlier title and scientific-code revision — 8 September 2026

The approved title is now **Discovering Molecular Substructures from Mass
Spectra with Neural Topic Models**, in both the printed heading and PDF
metadata. The first page was rendered and inspected; pages 2–14 are
pixel-identical to the accepted pre-title rendering. The final 14-page PDF
has SHA-256
`a38004e5f8876d3b271f921ce64536326b805c341c1322db7e797d9c7c7767a3`.

The [scientific code-quality review](code_quality_review_20260908.md) documents
functional simplifications, stronger input/evidence contracts and the corrected
uniform-evidence gradient. Its final checks report 232 scientific and 124
production tests passing, plus all 75 research Python files passing Black/Ruff.
The saved fits and numerical report values are unchanged, not newly retrained.
Only generator-source provenance changed in the three review summaries; all
24 generated artifacts are stable across repeated standard-library-only runs.

## Sharing revision before the discussion rewrite — 8 September 2026

The selected model, its nine saved fits and the historical evidence are
unchanged. Five new baseline fits completed successfully: three Tomotopy runs
and two plain-ETM runs. Together with the retained plain-ETM fit, these provide
three fits for every Table 4 row. All Tomotopy runs satisfied the unchanged
convergence rule after 750–800 iterations; none reached the iteration cap.

Independent aggregation and raw-array checks reproduce the reported means,
sample SDs, completion denominators and effective-topic diagnostics. Useful
motifs average 390.7 ± 2.1 for the selected model, 351.7 ± 7.0 for Tomotopy and
125.0 ± 6.1 for plain ETM. The manuscript states the accompanying trade-offs:
Tomotopy has higher conditional mean SOS; plain ETM has both lower completion
NLL and slightly higher conditional mean SOS than the selected model. These
are fixed-split development comparisons, not significance or generalization
claims. The plain-ETM batch-size and retained-fit environment differences
remain explicit.

The portable baseline bundle contains six fit records and 76 hash-verified
payloads, including its execution logs and separate training/postprocessing
source archives. Its manifest SHA-256 is
`8769a5ee7628119ae08bd3e23be4a43b25f896e95eba6dcd2aa2bfeea1913a35`.
All manifest-listed logs are part of the shareable evidence, despite the
repository's general log-file ignore rule. No selected-model fit was replaced,
and no real test-set input was opened for new fitting or evaluation in this pass.

Figures 1 and 3 became worked examples with a clear mathematical reading
order; at this checkpoint the report combined Discussion and conclusion.
The equation audit and follow-up thermo-nuclear review retain function-first
scientific code with explicit tensor dimensions and mathematical comments;
they strengthen input, cache and evidence-publication boundaries without
adding model components.

| Final check | Observed result |
| --- | --- |
| Canonical-environment research and production tests | 337 + 125 = 462 passed; two existing Lark deprecation warnings |
| Lean CPU environment matching focused CI dependencies | 336 passed; one expected CUDA-unavailable skip |
| Black and research-config Ruff | All 82 workflow-scoped Python files pass; the additional production MAG regression test also passes both checks |
| Four standard-library-only report generators | All 21 generated LaTeX fragments and three review summaries are byte-stable across repeated generation |
| Wheel build and isolated installed import | Pass; production CLI included, research package and runners excluded |
| MkDocs rebuild | Pass; existing production docstring and guide-link warnings remain visible |
| LaTeX build | 15 A4 pages; no warnings, undefined references/citations or overflowing boxes |
| Visual QA | All 15 final pages rendered and inspected, including all figures, tables and the compact reproducibility appendix |

The canonical PDF SHA-256 at that checkpoint is
`a740b745be1f4851072d7479ce3375185e8165c3ea4f87a0cf81a2e3e8253317`.
The historical 29-page PDF remains byte-identical. The selected-run CSV,
checkpoint-reload audit and historical validation comparison likewise retain
their original hashes. The pre-sharing checkpoint is preserved at
`backup/pre-sharing-revision-20260908`; the experimental branch is consolidated
into one commit above upstream revision `645a081`, without changing the fork's
reviewed `main` checkpoint or pushing any branch.

## Discussion rewrite and separate Conclusion — 8 September 2026

The requested editorial revision replaces the short, bold-led discussion
fragments with four substantive subsections: 5.1 Discovery breadth and
predictive fit; 5.2 Compact mixtures and the ETM foundation; 5.3 Limitations of
the current evidence; and 5.4 Implications for validation and future encoders.
Each develops its argument in two or three full prose paragraphs. Section 6
is a separate, concise Conclusion. The report-building and validation passes
preserve the scientific qualifications while connecting findings, interpretation
and implications within each subsection. No new chart or table is needed:
this is interpretation of the existing evidence, not an additional analysis.

At this checkpoint, only the Discussion/Conclusion passage changed in the LaTeX manuscript;
the source before it and from Data/code availability onward is exactly
unchanged. The source inventory, research README and manuscript-contract
test reflect the new hierarchy. The test checks numbered subsections,
multiple paragraphs, absence of bold paragraph headings, and Conclusion
before availability. All 337 research tests pass, and Black/Ruff pass on the
edited test. No scientific implementation or saved result changes.

All four report generators still produce the same 21 LaTeX fragments and
three review summaries. Baseline source hashes and the six-fit inventory
are reverified during generation. The rewritten interpretation retains the
conditional SOS and predictive-fit trade-offs, development-selection limits,
plain-ETM recipe/runtime caveats, and the need for independent chemical
validation. The assessment remains share with those caveats, not a new
generalization or component-causality claim.

The PDF remains 15 pages and compiles without warnings, undefined references
or overflowing boxes. All pages were rendered; pages 12–14 were visually
inspected afresh, and pages 1–11 and 15 are byte-identical PNGs to the previous
reviewed rendering. The PDF SHA-256 at that checkpoint is
`94a9a2dda3a9c6ea5b3d0191f2ee634cd67cc146dc6de5701136faf0882bc3d6`.
The prior commit is preserved at `backup/pre-discussion-revision-20260908`.
The existing experimental commit is amended rather than adding a second
upstream-relative commit; no push or production-model change is involved.

## Whole-manuscript language review — 8 September 2026

An independent review agent read the complete manuscript, appendix and figure
sources, then reviewed the revised prose in a second pass. The edits replace
abstract or reviewer-facing phrases with direct scientific descriptions,
unpack dense noun phrases, and connect methods, observations and implications.
For example, the proposed validation now specifies data not used for model
development or selection, evaluation criteria set in advance, and comparable
training and inference conditions. The four numbered Discussion subsections
and separate Conclusion are retained.

The [ETM paper](https://aclanthology.org/2020.tacl-1.29/) and
[Spec2Vec paper](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1008724)
were consulted for examples of direct methods-to-findings prose and discussion
flow. No passages were copied and no new scientific claims or bibliography
entries were added. The report-building and validation checks preserve the
conditional interpretation of SOS, the development-data selection caveat,
the plain-ETM training/runtime differences, and the distinction between a
combined-model comparison and evidence for individual components. The revised
discussion describes broad topic use as an observation, not an established
causal effect of shared embeddings.

All 13 displayed equation blocks, manuscript input paths, labels, citations,
bibliography, figure sources and result tables are unchanged from the
pre-language-review commit. No scientific code, saved fit or numerical evidence
was altered. All 337 research tests pass. The four standard-library-only report
generators reproduce the same 21 LaTeX fragments and three review summaries;
generation also verifies the six baseline fit records and 76 manifest-listed
payloads.

The final PDF has 15 A4 pages and compiles without warnings, undefined references
or overflowing boxes. Every page was rendered and visually checked during this
pass; pages 1–6 of the final rendering are pixel-identical to the proof already
inspected before the last line-wrap adjustment. The canonical PDF SHA-256 is
`7d6a070996ad53f24ad3175fb09728ef82125bd8fd91624c85740e48c201e464`.
The prior commit `f9d5e71` is preserved at
`backup/pre-language-review-20260908`. The existing experimental commit is
amended, keeping one commit above upstream revision `645a081`; the fork's
reviewed `main` remains unchanged and no branch is pushed.

## Automated chemical assessment — 8 September 2026

Four post-hoc assessments extend the existing development-validation results:
compound-level SOS excess over an acquisition/mass-matched background, explicit
MACCS-feature enrichment, motif stability and redundancy, and annotated MotifDB
correspondence with direct mass-overlap checks. The six existing selected-model
and Tomotopy fits are unchanged. Plain ETM remains in the original comparison;
its incomplete local full-fit inventory is not used for these additional tests.
The protocol and 64-path input seal precede the new calculations, but this is
not prospective registration or independent validation of a selected model.

| Report addition | Auditable source |
| --- | --- |
| Methods 3.11 and equations 14–15 | research/motif_chemical_assessment_20260908/protocol.json; benchmarks/neural_ms2lda/chemical_nulls.py |
| Results 4.3, Table 5 and Figure 4 | evidence/primary_50_topics.jsonl, primary_50_enrichment.npz and specificity_complete.json in the assessment directory |
| Background/scaffold sensitivity | evidence/sensitivity_25_*, sensitivity_100_* and scaffold_50_*; each configuration is evaluated separately |
| Results 4.4 and Figure 5A | evidence/topic_matches.jsonl, redundancy.jsonl and stability_summary.json |
| Results 4.5, Table 6 and Figure 5B | evidence/references.json, reference_hits.jsonl, reference_similarities.npz and references_complete.json |
| New generated tables, macros and figures | scripts/generate_motif_chemical_assessment_report.py; all inputs and completed-stage payload hashes are verified |
| Full replay and independent multiplicity checks | research/motif_chemical_assessment_20260908/validation.json |
| Scientific and strict maintainability review | docs/research/chemical_assessment_review_20260908.md |

The complete assessment was rerun in a separate output directory. All three
stage summaries and 27 scientific payload files reproduce exactly. Two
reference-score payloads differ by at most 3.33e-16 with different BLAS thread
counts; ranks, identities, support, direct overlaps and displayed values are
unchanged. Independent SciPy BY adjustment agrees on every full test family.
All 476 research/production tests pass, with two pre-existing Lark deprecation
warnings. Black, research-config Ruff and the wheel build pass. Whole-site
MkDocs was unavailable in this environment; the requested PDF was built and
checked independently. No site or navigation code is changed.

The original four report generators retain their 21 fragments and three
review summaries unchanged. The chemical-assessment generator adds three
LaTeX fragments, one summary and two static figures. The report preserves its
conventional structure, substantive numbered Discussion subsections and
separate Conclusion. Findings are explicitly conditional on development data:
both methods have modest excess SOS, the neural motifs have more repeatable
spectral patterns, and reference matches provide only partial corroboration.
Neither adjusted p-values nor illustrative reference hits establish chemical
novelty or expert-confirmed fragmentation assignments.

The final PDF has 19 A4 pages and compiles without warnings, undefined
references or overflowing boxes. Every page was rendered and inspected;
unchanged pages were compared with the reviewed proof, and changed pages were
inspected afresh. Its SHA-256 is
`3e3463c54b48c1900604a0eab79c351e2be27f1402c796f0e06bc40ca6b2d1fc`.
The preceding commit `cb41e38` is preserved at
`backup/pre-chemical-assessment-20260908`. This work amends the single
experimental commit above upstream revision `645a081`, without changing the
fork's reviewed `main`, retraining a model or pushing a branch.

## Directed correspondence with Tomotopy — 8 September 2026

The new primary-comparator section asks how much of each fixed motif inventory
has a counterpart in another fit, without requiring a one-to-one assignment.
Source and target cohorts are selected before matching; all source motifs
remain in spectral recovery denominators, including weak matches. The analysis
uses all 30 ordered comparisons between the same six saved fits, three cohorts,
two similarity definitions and a 201-point threshold grid. No model is retrained
or reselected, and no favorable chemical-correctness threshold is chosen.

| Current report location | Auditable source |
| --- | --- |
| Methods 3.12 and Equation 16 | research/cross_model_overlap_20260908/protocol.json; benchmarks/neural_ms2lda/cross_model_overlap.py |
| Results 4.3.1, Table 5 and Figure 4 | evidence/summary.json and inventory.json in the overlap directory; ordered-fit recovery and source-fit aggregation |
| Results 4.3.2 and Figure 5 | evidence/directed_matches.npy; full-beta, channel, supporting-compound and signed feature-effect agreement, with explicit missing-profile counts |
| Results 4.3.3 and Figure 6 | evidence/examples.json; fixed score-quantile selection, actual top-word probabilities and typed mass-overlap sensitivities |
| Generated overlap table, macros, summary and three figures | scripts/generate_cross_model_overlap_report.py; each payload hash is verified before report generation |
| Full byte-identical replay and independent coverage/matching audit | research/cross_model_overlap_20260908/validation.json |
| Scientific-code and strict maintainability review | docs/research/cross_model_overlap_review_20260908.md |

The added section is emphasized in Abstract, Discussion and Conclusion.
Cross-model mean median closest-counterpart cosine is about 0.454 over all
topics and 0.420/0.460 in the recurring, MAG-evaluable directions. These are
similarities, not recovered percentages. Repeat-run benchmarks are higher;
channel and cohort sensitivities matter, and much of the correspondence lies
in fragments rather than neutral losses. The earlier exploratory post-matching
filter retained a favorable subset of pairs, so its approximately 0.638 median
is not reported as coverage of every eligible source motif. Supporting compound
and feature agreement is partial, not independent chemical validation.

The four scientific payloads reproduce byte-for-byte in a fresh replay.
Independent sorted-score checks verify all 133,220 source rows and every
coverage denominator across 180 groups, with zero numerical difference.
Reciprocal flags and free-nearest-versus-eligible-Hungarian invariants pass.
Target fits are averaged within a source fit before reporting three source-fit
summaries; shared fits make the SDs dependent descriptive quantities, not
confidence intervals or nine independent cross-model replicates.

The full research and production suites pass **492 tests**, with two existing
Lark deprecation warnings. Workflow-scoped Black/Ruff check 102 Python files.
All 34 checked generated fragments, summaries and overlap figure assets are
byte-stable under regeneration, including the original model and chemical
results. The wheel builds and imports from the wheel itself, with all research
commands and data excluded. The package-content check caught and corrected
omitted exclusions for the preceding chemical commands as well as the new
overlap commands; a regression test now covers the scripts namespace. CI lint,
standard-library report generation and focused Numba dependencies are updated.
These local checks do not claim that remote CI or a whole-site MkDocs build ran.

The canonical PDF has **24 A4 pages**, with no LaTeX warnings, undefined
references/citations or overflowing boxes. Every page was rendered; pages
1--11 and 13 are pixel-identical to the reviewed proof, and every changed page
was inspected afresh. Its SHA-256 is
`2a37228fe5169503b478e904dd4c3629850c38a7cca3286faa8f51a8c131a2bc`.
The previous commit `b7cce23` is preserved at
`backup/pre-overlap-results-20260908`. This extension is consolidated into the
existing experimental commit above upstream revision `645a081`; the fork's
reviewed `main`, original model fits and preceding evidence remain unchanged.
No branch is pushed.


## Model exposition revision - 18 September 2026

The ETM equations and enhancement figures were still present in the 8 September
source and PDF. This revision improves their prominence and continuity rather
than recovering deleted material. The Tomotopy comparison now follows the
complete model. Methods 3.3 gives the ETM decoder, generator, approximate
posterior and base ELBO; 3.4-3.6 explain the three enhancements; 3.7 assembles
them; 3.8 compares the resulting components with Tomotopy. Evaluation remains
in 3.9-3.12. Adding the base ELBO shifts subsequent equation numbers by one:
the historical chemical-assessment map's equations 14-15 are now 15-16, and
the overlap map's equation 16 is now 17. Those dated maps describe earlier
revisions. Figures and tables retain their numbers.

### Exposition examples consulted

- [Dieng, Ruiz and Blei, Topic Modeling in Embedding Spaces (2020)](https://aclanthology.org/2020.tacl-1.29.pdf):
  Sections 3-5 establish the foundations, state the generative model, then
  develop inference and estimation. Figures 2-3 make topic embeddings concrete.
  Applied here: separate generation from inference, explain quantities before
  equations, and state the full base objective before modifying it.
- [Dieng, Ruiz and Blei, The Dynamic Embedded Topic Model (2019)](https://arxiv.org/pdf/1907.05545):
  Sections 3 and 4 distinguish the inherited ETM from its extension; 4.1 and
  4.2 distinguish the model from its inference algorithm. Applied here: identify
  exactly which three components change, then assemble the final model. This
  is an exposition reference, not an additional model or baseline in our study.
- [van der Hooft et al., Topic modeling for untargeted substructure exploration
  in metabolomics (2016), author manuscript](https://www.pure.ed.ac.uk/ws/portalfiles/portal/178956661/130100.pdf):
  Figure 1 explains the document/spectrum analogy; Figures 2-3 show motifs
  through spectral examples. Applied here: retain the spectrum-to-word figure
  and numerical mixture figure, and replace Figure 2b's abstract context boxes
  with explicit vectors for one word in two different spectrum contexts.

The papers guide organization and explanation; their wording and artwork are
not reproduced. The new examples are illustrative calculations from the stated
model equations. All experimental results, data, code and generated evidence
remain unchanged.

Validation: the final PDF has **25 A4 pages** and compiles without LaTeX
warnings, undefined references/citations, or over/underfull boxes. All pages
were rendered and inspected in contact sheets; the core model pages were also
inspected at full size. After the final Figure 2 label-spacing adjustment,
all 25 pages were rendered again: only page 6 changed, and it was inspected
again. The evaluation section and all subsequent source text are byte-identical
to the preceding revision; the Tomotopy comparison was moved intact. The new
context and mixture examples were checked numerically. `git diff --check` passes.
No model tests or training runs are needed for these manuscript-only changes.

Final PDF SHA-256: `a631bce95c6714d02f34e62b01ee620910439e9b57bea2756d137b2216beab43`.


## Main-paper and supplementary split - 18 September 2026

At the user's request, the main paper now keeps the scientific narrative and
five essential numbered equations, while the new supplement retains the full
mathematical specification, detailed evaluation protocols, supplementary
diagnostics and reproducibility record. The main paper is **17 pages**, down
from the preceding 25-page version; the supplement is **14 pages** including
its contents page and shared bibliography. All three teaching figures and
actual motif counterpart spectra remain in the main paper. The 17 original
numbered equations (16 equation/align environments) are preserved unchanged
in Supplementary S2 and S4-S6. All 41 original source labels survive across
the two documents. Generated result tables, figure data and fitted evidence
are unchanged.

The user-requested independent editor reviewed the allocation and both revised
sources against the ETM, Dynamic ETM and original MS2LDA examples. Its
clarity findings were applied and rechecked; no blocking editorial issues
remain. See editorial_review_20260918.md for the review and disposition.

Both documents compile without warnings, unresolved references/citations or
over/underfull boxes. All 31 pages were rendered and visually reviewed, with
full-size checks of the model narrative, explanatory figures, supplement
contents and technical page layout. After the final two clarity adjustments,
all pages were rendered again: only main pages 5 and 8 differed, and both were
inspected afresh. The supplement was pixel-identical to its reviewed proof.
PDF destination checks verify 24 main-to-supplement and 7 supplement-to-main
links, plus 90 internal named links, without broken destinations. Keep the
PDFs together for cross-document links; printed S-prefixed references remain
usable in viewers that do not open companion files. `git diff --check` passes.
This document-only revision does not retrain models or change numerical results.

- `contextual_sparse_etm_report.pdf`: SHA-256 `820b8cbef3a15757066b3a3d9929ae4d9ca68e346d801c49e812ac5189a30887`.
- `contextual_sparse_etm_supplement.pdf`: SHA-256 `66f4dd74a30678548b2a234cb120d2f128400adea3931a67c32b16c2dc32bb86`.

The PDFs and their sources are versioned together on the experimental branch;
this revision does not change the reviewed main branch or upstream repository.

## Final scientific and editorial audit - 18 September 2026

Four reviewer agents covered complete-paper editing, equation-to-code
alignment, numerical evidence and ML scientific soundness. Their corrections
clarify earlier test-set use, training versus validation, historical K=36
implementation limits, plug-in prediction, fixed-embedding repeatability,
pseudo-count weighting, hard routing and synthetic vocabulary exclusions.
Both documents now link the public research branch. Detailed findings and
verification are recorded in [the final scientific review](final_scientific_review_20260918.md)
and [the results audit](final_results_audit_20260918.md).

All 367 scientific and 125 production tests pass. The six report generators
produce no drift, and checkpoint/evidence replay agrees at the stated
numerical precision. No fitted parameters, generated result tables or
scientific executable algorithms changed in this audit. Two outdated
document-location contracts and equation-location docstrings were repaired.
CI now compiles both PDFs and checks their cross-references.

The final main paper remains **17 pages**, and the supplement **14 pages**.
All 31 pages were rendered and inspected. Compilation has no warnings,
undefined references/citations or over/underfull boxes. All 25 main-to-supplement,
7 supplement-to-main and 91 internal links resolve. Keep the PDFs together
for companion-file links. The five-equation main exposition and the complete
supplementary formulation remain intact.

- `contextual_sparse_etm_report.pdf`: SHA-256 `464d6ed6dfa1697aa7c2091fb416e25551a20bf946f949d1c909fd6b1c081387`.
- `contextual_sparse_etm_supplement.pdf`: SHA-256 `e27081eb8754972d598a166180a11c6d52f4050059b35524103de9d68e265c0d`.

The verdict is share with caveats as a development study, rather than
independent confirmation of chemical superiority. Exact historical K=36
training replay, new evaluation data and structural validation remain outside
this audit; the paper states those evidence limits explicitly.

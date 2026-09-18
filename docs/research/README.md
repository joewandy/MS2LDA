# Discovering Molecular Substructures from Mass Spectra with Neural Topic Models

[contextual_sparse_etm_report.tex](contextual_sparse_etm_report.tex) is the
canonical main manuscript; its matching PDF is the reviewed rendering.
[contextual_sparse_etm_supplement.tex](contextual_sparse_etm_supplement.tex) and
[its PDF](contextual_sparse_etm_supplement.pdf) hold the detailed methods,
mathematics, diagnostics and reproducibility record. Keep the two PDFs together:
numbered cross-document references link to the corresponding sections and figures.
Both sources use [a shared bibliography](contextual_sparse_etm_references.tex).

The main report describes **one chosen model**, with **Tomotopy LDA as the
primary comparator**. A plain ETM reference appears in the main comparison
table as a secondary check of what the enhancement package adds. The report
is not a model-search diary.

The existing scientific structure is retained: Abstract, Introduction,
Related work, Materials and methods, Results, Discussion, a separate
Conclusion, short Data/code availability and References. Reproducibility
details now appear in Supplementary Section S8. Results clearly separate synthetic recovery from MSnLib validation.
Discussion uses four substantive numbered subsections: discovery breadth,
chemical specificity and predictive fit; compact mixtures and the ETM foundation;
evidence limitations;
and future validation and encoders. It does not use bold paragraph-level headings.
Methods explain the published ETM base (decoder, generator, Gaussian posterior
and the meaning of its training objective), followed immediately by three enhancements, each
introduced through its purpose and a worked example:

- equal fragment and neutral-loss emission mass;
- whole-spectrum contextual evidence correcting the Gaussian posterior mean;
- 1.5-entmax topic mixtures.

The full Tomotopy component comparison is Supplementary Table S2, referenced
from the main comparison-design section. Figure 2 includes numerical
channel and sparsity examples and a geometric example of the same fragment
receiving different context in two spectra.

The conventional two-hidden-layer MLP is retained. The selected implementation is
reduced_document_context in
[contextual_reductions.py](../../benchmarks/neural_ms2lda/contextual_reductions.py).
The frozen leave-one-out artifacts and production Tomotopy backend are
unchanged by this review. A shared uniform-evidence gradient correction affects
future training, not the saved results; see the code-review record below.

The main paper retains the worked preprocessing example, three-panel
enhancement illustration, numerical mixture calculation, plain-language metric
explanations and essential evidence caveats. The full notation table, LDA/ETM
component mapping, expanded objectives and mathematical evaluation definitions
are in Supplementary Sections S1-S6. Additional correspondence diagnostics are
in S7; the former reproducibility appendix and component-evidence scope are in S8.
The independent [editorial review](editorial_review_20260918.md) records the
allocation and final clarity checks.
The subsequent [final scientific review](final_scientific_review_20260918.md)
records the complete editorial, ML-methodology and code-alignment audit;
the [results audit](final_results_audit_20260918.md) records numerical replay
and its limits. All review corrections are reflected in both PDFs.
Figures 1 and 3 now use a concrete spectrum-to-token example and a numerical
topic-mixture calculation instead of box-and-arrow wiring diagrams.
The [code-quality review](code_quality_review_20260908.md) records the subsequent
scientific-code audit, regression checks and equation-to-code documentation.
The [equation audit](equation_code_audit_20260908.md) records the independent
end-to-end calculation, numerical conventions, and evaluation edge cases.

## Evidence boundary

All current-model results come from six synthetic and three real-validation
fits. The main validation table compares three fits of each model, with means and sample SDs
for every metric. Tomotopy uses three fresh fits with saved mixture arrays;
the historical single fit is not pooled as a fourth. Plain ETM retains its
original fit and adds two initializations using the same batch-256 recipe,
versus batch 200 for the selected model. No selected-model fit is rerun.
Standard deviations describe between-fit training variation on fixed data,
not uncertainty over new chemical populations or individual motif identities.
The retained plain-ETM fit used WSL2 and its new repetitions native Linux;
scientific package versions match, but the SD is not purely seed-only variation.
The best observed directional means are bold; effective topic count is
descriptive and is not ranked. Bold type is not a significance claim.

These are development-validation results, not an independent post-selection
test. Historical test numbers are not transferred to the chosen architecture.

## Automated chemical assessment

The report now also assesses matched-background SOS, explicit MACCS feature
enrichment, motif repeatability/redundancy, and correspondence with annotated
MotifDB references. These use the existing three selected-model and three
Tomotopy fits without retraining. Plain ETM is not added to these new analyses
because one full repeat fit is unavailable locally; its original three-fit
main comparison is unchanged.

The findings narrow the chemical claim: both models have a similar modest
SOS excess above matched background; much of the additional evaluable neural
inventory has single-compound support. Neural spectral patterns are more
repeatable, but their supporting compound groups are not reliably identical.
Most MotifDB similarities are modest and even strong examples can explain
only part of a motif. These are candidates for chemical follow-up, not a
count of confirmed new substructures.

The [assessment protocol and evidence guide](../../research/motif_chemical_assessment_20260908/README.md)
defines all four analyses, frozen settings, output axes and reproducibility
commands. The [assessment validation review](chemical_assessment_review_20260908.md)
records the strict scientific-code review, exact-null tests, full replay and
report checks. These analyses are introduced in the main methods, reported in Results
4.4--4.6, and fully defined in Supplementary Sections S4-S5.

## Correspondence with Tomotopy

Dedicated Results Section 4.3 and its recovery table and spectral figures ask whether comparable
aggregate scores correspond to the same spectral motifs. The result is
**partial correspondence, not chemical equivalence**: closest cross-model
matches fall below repeat-run agreement, and fragments agree much more
strongly than neutral losses. Supporting compound groups often differ.
This key result is also reflected in the Abstract, Discussion and Conclusion.

The analysis filters both candidate inventories before matching, permits
target reuse, and retains every source motif in the recovery denominator.
It evaluates both directions over the full similarity range; it does not
choose a threshold to claim a favorable percentage of chemically recovered
motifs. All-topic and recurrence-only sensitivities accompany the recurring,
MAG-evaluable comparison. Main figures show recovery against repeat runs and
actual paired motif spectra; Supplementary Figure S1 shows chemical agreement,
multiplicity and channel sensitivity.
Examples follow a fixed score-quantile rule, not chemical or visual selection.

The [overlap protocol and evidence guide](../../research/cross_model_overlap_20260908/README.md)
documents the implementation and reproduction commands. The
[scientific-code and validation review](cross_model_overlap_review_20260908.md)
records full byte-identical replay, independent denominator checks, and
the important distinction from filtering only successful pairs after matching.
No model was retrained and the original comparison results are unchanged.

## Preserved model-selection record

The full experimental history and completed ablations remain in
[the research review](../../research/minimal_neural_etm/review_20260907/README.md).
The pre-refocus 29-page report is archived there as:

- model_form_review_report_20260908.tex
- model_form_review_report_20260908.pdf
- model_form_review_report_sources_20260908.md

This preserves the previous detailed audit without carrying its variants or
historical test results into the focused manuscript.

## Regeneration

From the repository root, using the ms2lda-neural environment:

    python -S -m scripts.generate_contextual_reduction_review
    python -m scripts.generate_motif_chemical_assessment_report
    python -m scripts.generate_cross_model_overlap_report
    pytest -q benchmarks/neural_ms2lda/tests

The generator validates the complete research inventory, then separately
requires exactly the chosen model's nine fits and six declared baseline fits.
The [baseline evidence](../../research/baseline_repeats_20260908/evidence/)
is checked against its source hashes and fit identities before tables are
written. It generates the four
current_contextual_etm fragments in docs/research/generated: macros,
validation comparison, seed stability and synthetic results. The main
validation table contains Tomotopy, plain ETM and the selected model, with
the number of fits stated explicitly. The per-seed stability fragment is
retained for audit but is not included in the manuscript. Main-text stability
uses aggregate values; exact random states appear in Supplementary Section S8.
The chemical-assessment generator independently verifies its evidence bundle
before creating the new table, macros, reference examples and two figures.
The overlap generator likewise verifies its own sealed bundle before creating
its table, macros, summary and three figures; raw fitted matrices are not
needed to regenerate the report from the committed evidence.
Use `--tables-only` with `python -S` when only standard-library dependencies
are available; figure generation uses Matplotlib and NumPy.

The historical and full-review generators are retained for the archived
audit and are not inputs to the main manuscript:

    python -S -m scripts.generate_contextual_sparse_etm_report
    python -S -m scripts.generate_minimal_etm_review
    python -S -m scripts.generate_published_neural_review

Compile both documents with a current TeX Live installation, using a separate
build directory and making its auxiliary files available to `xr-hyper`. From
`docs/research`, the following alternates three passes per document so links
work in both directions (the build directory must not contain source backups):

```bash
report_source_dir="$PWD"
report_build_dir=$(mktemp -d)
for report_pass in 1 2 3; do
  for report_name in contextual_sparse_etm_supplement contextual_sparse_etm_report; do
    TEXINPUTS="$report_build_dir:${TEXINPUTS:-}" pdflatex \
      -interaction=nonstopmode -halt-on-error \
      -output-directory="$report_build_dir" "$report_source_dir/$report_name.tex"
  done
done
```

Check the final logs for unresolved references or layout warnings; render and
inspect every page of both PDFs before replacing the canonical pair. Relative
PDF links are designed for the two files to be distributed together. Some PDF
viewers require permission to open a companion document; printed section and
figure numbers remain usable. The archived report can still be compiled from
`docs/research` so its `generated/` references resolve.

[Source inventory and figure/table map](contextual_sparse_etm_report_sources.md)
document provenance, definitions and report QA. The sharing revision adds five
baseline fits, not new selected-model fits, test-set evidence or production
neural integration. The underlying research is consolidated in one upstream-based commit. The
main paper, supplement, shared bibliography and editorial review are versioned
together on the experimental branch. Publication targets the user's fork;
the reviewed main branch and upstream repository are unchanged.

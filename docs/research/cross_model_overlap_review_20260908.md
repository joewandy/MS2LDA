# Directed Tomotopy correspondence: scientific-code and reporting review

This extension evaluates the existing six fitted inventories. It does not
train or select a model, change production Tomotopy, or alter the preceding
chemical assessment. The dedicated report section describes correspondence
with the primary comparator, not a catalogue of additional model variants.

## Scientific question and denominator correction

The question is whether **every motif in a defined source inventory** has a
spectrally similar counterpart in a target fit. Each source and target cohort
is selected before matching. Target reuse is allowed; weak best scores remain
in the source denominator. All ordered fit pairs are used, without choosing
the best target run or combining target runs into a larger search library.

An earlier exploratory comparison kept only Hungarian pairs whose two members
were recurring and MAG-evaluable after matching. Its higher median (about
0.638 across models) describes that selected subset, not inventory coverage.
The present recurring/evaluable directed summaries are about 0.420 and 0.460.
These numbers are not contradictory: eligibility and the denominator changed.
The main manuscript explains why filtering successful pairs after matching
can conceal difficult source motifs. It does not present the earlier favorable
subset as the recovery of the complete eligible inventory.

## Equation-to-code audit

| Mathematical operation | Implementation and check |
| --- | --- |
| Eligible topic-index sets in Equation 16 | `cohort_topics` independently selects source and target before `directed_records` slices the similarity matrix; regression example gives an ineligible nearest target |
| Full-beta cosine on common coordinates | `spectral_vectors` normalizes all ordered vocabulary coordinates; `similarity_matrices` takes inner products; seals require byte-identical vocabularies |
| Fragment/loss sensitivity | Channel vectors are independently normalized; arithmetic mean gives the two cosines equal weight; a typed-fragment/loss test prevents equal masses becoming the same coordinate |
| Best and second-largest score | `nearest_two` permits reuse and breaks exact ties by ascending topic ID; a single target has no second neighbor, represented by ID -1 and missing score |
| Coverage C(t) and multiplicity M(t) | `recovery_curves` divides inclusive threshold counts by the full source cohort; zero/exactly-one/multiple fractions are 1-C, C-M, M |
| Supporting compounds and signed feature effects | `directed_records` computes diagnostics only after choosing a spectral target; effects are the count-minus-matched-expectation prevalence differences from Equation 15 |
| Missing quantities | Undefined union/profile cosines remain missing; no missing annotation is silently assigned zero chemical agreement; zero-valued available fingerprints remain available |
| Aggregation | `summarize_matches` computes pair medians, averages target fits within each source fit, then reports three source-fit summaries; shared-fit dependence is stated |
| Representative score positions | `quantile_example` uses a frozen score-quantile rule and stable identity tie-break; chemical agreement never affects example selection |
| Displayed words and direct mass overlap | `example_overlap` uses typed top-20 words, original probabilities and maximum-cardinality one-to-one mass matches at each declared tolerance |

Both channels have positive mass in all six saved fits. Invalid/nonfinite
spectral vectors fail rather than acquiring fictional scores. The overlap
protocol is post-hoc on development data, fixed after earlier correspondence
results were seen, not a prospective registration or independent confirmation.

## Strict maintainability review

The thermo-nuclear review favors short, documented mathematical functions.
No classes, abstract base types, plugin registries or generic workflow engine
were added. Five research-only modules separate numerical operations, sealed
evidence, examples, report aggregation and figures; three small command-line
entry points run, regenerate and audit the analysis. These are distinct
scientific responsibilities, not alternative implementations of the maths.
Existing normalization, mass matching, JSON, hashing and atomic-array helpers
are reused. The report-only path remains standard-library compatible.

Input and publication boundaries reject a changed protocol, changed inputs,
nonempty seal destination, incomplete payload manifest and incomplete fit
inventory. Protocol-list order is checked because numeric array codes depend
on it. Completion follows all payload writes and a second input-hash check.
The compact array contains numeric fields only and is loaded without pickles.
No blocking correctness or maintainability issue remains in this scoped pass.

The wheel-content audit caught six research commands leaking into the
production package: the three new overlap commands and the three preceding
chemical-assessment commands. Explicit package exclusions now keep them
clone-runnable only. A regression test covers every nonproduction command
in the scripts directory; CI lint and report-regeneration steps also include
the new analysis, and the focused dependency install includes the existing
chemical assessment's Numba requirement. No production modeling or annotation
function changes.

## Numerical replay and independent checks

The [machine-readable audit](../../research/cross_model_overlap_20260908/validation.json)
records byte-identical replay of all four scientific payloads in a fresh
directory. The audit separately checks:

- All 133,220 directed rows across 180 fit-pair/cohort/similarity groups.
- Every source denominator and every recovery/multiplicity curve using sorted
  scores and `searchsorted`, independent of the runner's threshold matrix;
  maximum difference is exactly zero.
- All 133,220 reciprocal flags against the corresponding reverse search.
- Free nearest scores are at least the earlier Hungarian scores whenever
  the forced mate is eligible: 30,000 all-topic and 9,834 recurring/evaluable
  comparisons. This invariant does not filter source motifs from new curves.
- Small-matrix exhaustive threshold-graph, tie, reuse, one-target, missing
  profile, aggregation, protocol-code and evidence-boundary regressions.

The full research and production test suites pass: **492 tests**, with two
pre-existing Lark deprecation warnings. Black and the research Ruff configuration
pass. Regeneration checks cover both original report fragments and the new
evidence-backed table, macros and static figures. Build and final PDF checks
are recorded in the accompanying manuscript source inventory.

## Interpretation and visual safeguards

Results Section 4.3, Table 5 and Figures 4--6 make the overlap question prominent.
The recovery figure shows all source motifs against the full threshold range,
alongside repeat-run benchmarks. Thin traces are source-fit variation, not
confidence intervals. Chemical-density panels share axes and a color scale;
they count dependent fit-pair/topic records and disclose undefined profiles.
The example figure uses actual probability-stick plots with separately typed
channels, consistent mass axes and within-topic maximum scaling. These are
probabilities, not measured intensities. All pages are rendered and inspected
before the canonical PDF is replaced.

The supported conclusion is partial spectral continuity with Tomotopy, often
concentrated in fragments. Compound and chemical-feature agreement is modest;
high spectral cosine can reflect one dominant fragment rather than the whole
motif. No selected similarity cutoff, chemical-equivalence claim, additional
novel-substructure count, split/merge proof or unmeasured speed claim is added.

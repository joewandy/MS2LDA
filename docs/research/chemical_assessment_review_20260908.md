# Chemical-assessment scientific and code-quality review

## Scope and outcome

Implemented the four approved post-hoc assessments, preserving the selected
model, all saved fits, the production Tomotopy path and the original comparison
tables. This is a **conditional development-set assessment**, not independent
confirmation or evidence of chemical novelty. The report reflects the weak
as well as positive findings. No trial, annotation or motif was discarded to
improve the reported outcome.

The protocol and 64-path input seal predate the new outcome calculations.
The original seal is immutable. Its local protocol mirror was initially
reformatted JSON; validation restores a byte-identical copy of the original
sealed protocol after checking both semantic equality and the original hash.
No setting or seal is changed. New seals require an empty output directory
and preserve the protocol's exact bytes from the outset.

## Equation and data checks

| Scientific quantity | Code and check |
| --- | --- |
| Existing SOS | All six complete saved results reproduce exactly with the canonical scorer; independent integer-intersection reconstruction agrees within 1e-14 |
| Compound associations | Exact raw-USI metadata join for all 3,889 spectra; 3,854 connectivity compounds; duplicate fingerprints/scaffolds must agree |
| Conditional SOS expectation, Eq. 14 | `chemical_nulls.sos_permutations`: stratum population means weighted by the motif's compound counts |
| Permutation test | Complete fingerprints move together; all six fits share each draw; integer counts preserve inclusive ties; plus-one p-values never become zero |
| Null variance | Finite-population sampling without replacement, including the zero-variance case when an entire stratum is selected |
| Feature prevalence difference, Eq. 15 | `chemical_nulls.feature_enrichment`: observed count minus conditional expected count, divided by distinct compound support |
| Feature count test | Exact finite-support hypergeometric convolution; exhaustive small-population oracle agrees |
| Multiplicity | All 6,000 SOS and 996,000 feature slots per configuration; independent SciPy BY checks on every saved family |
| Sensitivities | 25/50/100 Da mass definitions and a separately rebuilt 2,856-compound scaffold-balanced cohort; no outcome-driven representative selection |
| Topic matching | Full common-vocabulary beta cosine, Hungarian mapping, explicit reciprocal tie rule; identity/reordering/duplication/orthogonality fixtures |
| Reference matching | Typed fragments/losses remain separate; full manual-label provenance and conflicts; missing embeddings cannot masquerade as hits |
| Direct overlap | Maximum-cardinality one-to-one mass matching agrees with a bipartite assignment oracle; decimal tolerance boundaries tested |
| Reporting | Exact protocol and payload hashes, six-fit inventory, fixed test-family/permutation counts; corrupted payload and reduced-budget fixtures fail closed |

The primary background has 158 strata and 64 fixed singleton compounds.
Null variability, Monte Carlo tail-probability precision, three-fit SDs and
background sensitivity remain separate quantities. BY correction cannot
repair selection bias or guarantee exchangeability among related compounds.

## Strict maintainability review

The thermo-nuclear review was applied to the new research-only code and its
interfaces with the previously reviewed branch. No production refactor or
additional class hierarchy was introduced. Statistical operations are small
functions with explicit topic/compound/feature dimensions and equation-level
docstrings. The largest new scientific source module is below 450 lines.

The main simplification is to precompute **integer fingerprint intersections**
and stream permutation totals. This removes both floating-point tie tolerances
and a large permutation-score storage layer. The only compiled kernel is a
short numerical loop; no model or analysis framework is introduced. The exact
feature test reuses SciPy hypergeometric probabilities; the existing MAG,
Spec2Vec, spectrum construction and atomic JSON writers remain canonical.
One unnecessary reference-stage forwarding wrapper was removed. Validation
is separated into small protocol, payload, summary and multiplicity checks.

Issues caught and corrected during implementation:

- The local reference files pad the absent channel with paired JSON `NaN`,
  not only `null`. The parser accepts paired missing padding, but rejects
  unpaired missing values, infinity and invalid intensities. No reference is
  silently repaired into a new spectrum.
- Native MACCS definitions are exposed as `smartsPatts`; an export regression
  now checks all 166 positions and procedural exceptions.
- Some saved beta arrays are Fortran-ordered float32. A float32 reduction over
  21,233 words can accumulate about 1e-4 summation error despite correctly
  normalized stored probabilities. Validation now sums in float64 instead of
  loosening tolerances or changing model values; a dedicated regression checks
  this and still rejects genuinely unnormalized rows.
- Reference embeddings are explicitly checked for finite nonzero norm, avoiding
  the production normalization helper's zero-division behavior without changing
  that production path.
- Scientific manuscript-contract tests now permit precisely the three new
  evidence fragments, while retaining the fixed model, conventional section
  order, no historical-test imports and seed-free main-text constraints.

No unresolved scientific-code blocker was identified in this scope. This is
not a claim that every historical or upstream production file is issue-free;
the preceding branch-wide reviews remain in the existing review documents.

## Full replay and shareability checks

The full assessment was recalculated into a separate output directory with
the same sealed inputs and all 99,999 permutations per configuration.
`research/motif_chemical_assessment_20260908/validation.json` records the
machine-readable comparison and source hashes.

- All three stage summaries are exactly identical.
- 27 scientific payload files are byte-identical, including every SOS and
  enrichment result, cohort, motif inventory, fit-pair match and redundancy
  result.
- The two reference-score payloads differ by at most 3.33e-16 because the
  reference runs used two versus four BLAS threads. Hit identities, ranks,
  support counts, all direct overlaps and all summary values are identical.
- Independent SciPy BY adjustment agrees for every full family.
- All 476 research/production tests pass; only two pre-existing Lark
  deprecation warnings remain. Black and the research Ruff configuration pass.
- Wheel build/import checks pass. Whole-site MkDocs was unavailable in the
  environment; no site or navigation code was changed. The requested LaTeX/PDF
  build is checked independently.

The main report includes one compact comparison table, one illustrative
reference table, two distribution figures and explicitly defined metrics.
Full fit-level identities and exhaustive outcomes remain in the evidence.
The final PDF is inspected in its native page layout, not only through text
extraction; the source inventory records its final hash and page count.

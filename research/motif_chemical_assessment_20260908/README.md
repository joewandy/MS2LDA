# Automated chemical assessment of the selected model

This is a post-hoc assessment of **existing development-validation fits**, not
an independent chemical validation or an expert annotation exercise. The model,
training, topic count, preprocessing and saved MAG annotations are unchanged.
The primary comparison is the selected Contextual Sparse ETM versus Tomotopy,
three saved fits each. Plain ETM remains in the existing main comparison but is
not included here: only two of its three full repeat fits are locally available.

## Frozen protocol and reproduction

`protocol.json` was written and `evidence/input_seal.json` was sealed before
calculating these new outcomes. The seal records all input hashes, bytes and
scientific dependency versions. The source implementation is separately hashed
in completed-stage records. Protocol settings are not a claim of prospective
registration: earlier development had already used these validation data.

Run from the repository root in the canonical `ms2lda-neural` environment:

```bash
python -m scripts.run_motif_chemical_assessment \
  --protocol research/motif_chemical_assessment_20260908/protocol.json \
  --output output/benchmarks/motif_chemical_assessment_reproduction --seal
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.run_motif_chemical_assessment \
  --protocol research/motif_chemical_assessment_20260908/protocol.json \
  --output output/benchmarks/motif_chemical_assessment_reproduction
```

The runner never overwrites an input seal or retrains a model. Completed stages
are resumed only after verifying their payload hashes. Raw saved fits and the
Spec2Vec model are large local inputs; the compact assessment evidence is tracked.
Use a new output directory for a full recalculation. The scientific payloads are
deterministic with the sealed dependencies and batch size, apart from
machine-precision reference-cosine rounding with different BLAS thread counts;
timestamped sealing metadata is intentionally not byte-identical on a second run.

Audit a complete replay and independently recompute all multiplicity corrections:

```bash
python -m scripts.validate_motif_chemical_assessment \
  --evidence research/motif_chemical_assessment_20260908/evidence \
  --replay output/benchmarks/motif_chemical_assessment_reproduction \
  --protocol research/motif_chemical_assessment_20260908/protocol.json \
  --output output/benchmarks/motif_chemical_assessment_reproduction_validation.json
```

The completed full replay is recorded in [validation.json](validation.json).
Its stage summaries and 27 scientific payloads are exact; the two reference
score files differ by at most 3.33e-16, with all ranks and overlaps unchanged.
See the [scientific/code review](../../docs/research/chemical_assessment_review_20260908.md)
for the checks, limitations and resolved implementation issues.

## Four assessments

1. **SOS excess above matched background.** Fixed MAG consensus fingerprints
   are compared with the fingerprints of motif-supporting compounds. Whole
   fingerprints are shuffled within acquisition-profile / precursor-mass
   strata, with the same draws across six fits. The 99,999 draws use inclusive
   integer intersection-count ties and the plus-one permutation p-value.
2. **Explicit feature enrichment.** All 166 native RDKit MACCS positions are
   assessed, not only the ones favored by MAG. Conditional stratum-specific
   hypergeometric distributions are convolved exactly. Prevalence differences
   are effect sizes; enrichment ratios with zero background are undefined.
3. **Stability and redundancy.** All vocabulary columns enter beta cosine and
   Hungarian matching. Reciprocal-nearest matches, top-20 Jaccard, compound
   Jaccard and signed feature-effect cosine provide complementary descriptions.
   Within-fit neighbours and repeated MAG fingerprints identify redundancy.
4. **MotifDB correspondence.** GNPS, MassBank and Urine positive reference sets
   are deduplicated by exact normalized typed spectral signatures. Manual labels
   and conflicts are preserved. Existing Spec2Vec matching is complemented by
   separate fragment/loss overlap at 0.01 Da and two tolerance sensitivities.

Each spectrum has one dominant topic; connectivity compounds are deduplicated
within that topic, not independently resampled for each spectrum. A compound
with several spectra may support several topics. Every topic remains in the
inventory. Missing MAG consensus differs from a valid all-zero fingerprint.
Singleton permutation strata remain fixed and degenerate nulls are flagged.

Primary strata use 50 Da bins. The two mass sensitivities use 25 and 100 Da;
a separate 50 Da sensitivity retains one hash-selected compound per nonempty
scaffold (acyclic compounds remain distinct). Selection uses no model score.
Benjamini--Yekutieli correction includes all 6,000 SOS slots and, separately,
all 996,000 motif-feature slots **in each fixed configuration**. Missing or
constant tests enter as p=1. Sensitivities are not pooled or selected to improve
significance. The q=0.05 count is a descriptive screen, not selection-adjusted
false-discovery control for this development study.

## Evidence schema

- `compounds.jsonl`: global compound indices follow file order; acquisition
  profiles, scaffolds, source spectrum IDs and pinned MACCS fingerprints.
- `inventory/<fit>.jsonl`: all topic IDs, global supporting compound IDs and
  fixed MAG consensus fingerprints (null means unavailable).
- `<configuration>_topics.jsonl`: SOS effect, support, null variability,
  Monte Carlo counts/precision, exploratory p/q and unevaluable reasons.
- `<configuration>_enrichment.npz`: rows concatenate fits in protocol order,
  then topics 0..999; columns are native MACCS positions 1..166. `count`,
  `support`, `expected_count`, `informative`, `p`, `q` suffice to reconstruct
  prevalence=count/support, background=expected_count/support, difference and
  ratio. Empty support gives undefined prevalence; zero expected count gives
  undefined ratio. Do not interpret those missing quantities as chemical zeros.
- `<configuration>_cohort.json`: global retained compound IDs and strata as
  **local** indices into the retained list.
- `maccs_definitions.json`: SMARTS/count thresholds and native procedural keys;
  position 0 is unused. MACCS keys are overlapping feature predicates, not 166
  unique substructures or unambiguous fragment identities.
- `topic_matches.jsonl`, `redundancy.jsonl`: complete matched/nearest-topic
  records. Fifteen pair comparisons are correlated, not 15 independent fits.
- `references.json`: reference signatures, full manual-label provenance and
  parse/embedding exclusions. `auto_annotation` is never used as truth.
- `reference_hits.jsonl`: five ranked references per scorable topic, typed
  direct overlaps and both denominators at each tolerance.
- `reference_similarities.npz`: full ranked-search information; rows follow
  fit/topic order, columns follow `references.json`. Unscorable values are NaN.
- `*_complete.json`: summaries and payload hashes. The report reads this
  verified evidence; no outcome is manually transcribed into a table.

## Interpretation and presentation contract

Report positive, weak and negative outcomes. Neither these conditional nulls
nor repeated fits remove selection bias. Conditioning on acquisition profile
and precursor mass also changes the question: these tests ask about structure
**beyond those attributes**, not all chemistry carried by molecular mass.
Uncontrolled structure/acquisition dependence can remain within a stratum.

The two figures use static, reproducible Matplotlib output in the existing
LaTeX/PDF report, with blue/solid selected-model and orange/dashed Tomotopy
encoding plus direct labels/legends. Their contracts are:

| Figure | Question and data | Form / expected rows | Interpretation |
| --- | --- | --- | --- |
| Chemical specificity | How large is SOS excess, and how much compound support underlies it? | Two empirical distributions, each fit shown separately; 6,000 topic inventory rows with explicit eligible/support subsets | Show full spread and small-support caveat, not only q-threshold counts |
| Spectral correspondence | How repeatable are motifs, and how closely do they resemble annotated references? | Full-beta matched-score ECDFs for 15 dependent pairs; best-reference similarity/coverage curves for six fits | Forced matches and embedding similarities are descriptions, not chemical correctness probabilities |

Chart families are distribution and threshold-coverage curves (not temporal
trends). Shared axes retain their natural [-1,1] or [0,1] metric scale. No
blossom/branding is added to this third-party scientific manuscript. Final QA
is inspection of the figures inside the rendered canonical PDF.

The compact comparison table reports fit-level means and sample SDs. Any
reference examples are explicitly illustrative: rank selected-model top hits
by Spec2Vec cosine, require at least two supporting compounds and nonzero
direct overlap, and take distinct reference signatures. This selection is
for inspection, not an unbiased sample or a new discovery threshold. Exact
fit/topic identities remain in the evidence rather than suggesting matched
topic IDs across runs.

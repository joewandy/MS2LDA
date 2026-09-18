# Cross-model motif overlap

This descriptive extension asks how much of each fixed motif inventory has a
counterpart in the other model. It uses the same three selected-model and three
Tomotopy fits as the chemical assessment, without changing or retraining them.
The analysis is post-hoc on development-validation data. Existing Hungarian
matches and the earlier exploratory filtered comparison remain historical
evidence; the new recovery analysis filters candidate inventories **before**
matching and permits target reuse.

## Fixed questions and denominators

- For each source motif, find its closest target within one target fit. Repeat
  for all ordered fit pairs, without pooling target runs into a larger library.
- Plot the fraction of every source inventory recovered across the entire
  cosine threshold range. Compare cross-model curves with repeat-run curves.
- Show all topics and recurring, MAG-evaluable topics. A third cohort drops the
  MAG requirement to check sensitivity to annotation availability.
- Check full-beta cosine against equal-weight fragment/loss cosine. On each
  selected pair also inspect channel-specific cosine, top-word overlap,
  supporting compounds and signed chemical-feature enrichment effects.
- Describe multiple candidate counterparts through the second-best score.
  Multiple spectral neighbours do not establish that a chemical motif has
  split or merged, nor that an unmatched motif represents novel chemistry.

The protocol was fixed before these extended calculations, after the existing
one-to-one results had been examined. It is not a prospective registration.
The input seal hashes the protocol and 23 used inputs, including the preceding
assessment and saved matrices. Completed evidence is separate from the
preceding assessment bundle; sealing rejects a nonempty output directory.

## Report and chart contracts

The existing LaTeX/PDF manuscript is the sole reader-facing report. Preserve
its title, conventional section structure, original model and result tables.
Methods 3.12 defines the metrics and Equation 16. Results 4.3, Table 5 and
Figures 4--6 present the comparison, with the supported conclusion reflected
in Abstract, Discussion and Conclusion. Implementation details stay here,
not in a seed-by-seed manuscript narrative.

| Figure | Question / form | Data and denominator | Interpretation |
| --- | --- | --- | --- |
| Bidirectional recovery | Four coverage-curve panels: each direction, all and recurring/evaluable | 30 ordered fit comparisons, 201 thresholds; three source-fit curves after averaging target fits | Cross-model recovery compared with within-model repeatability, not a validated chemical threshold |
| Agreement and multiplicity | Two-dimensional spectral/feature-effect distribution and second-neighbour coverage | Recurring/evaluable directed best matches, explicit undefined-chemical counts; all source topics remain in multiplicity denominator | Spectral agreement need not imply chemical agreement; target reuse is not proof of splitting |
| Matched spectral examples | Paired fragment/loss probability-stick plots, plus a one-to-two example | Deterministic score-quantile examples selected by the protocol; original typed top-20 beta words | Concrete pattern correspondence and counterexamples, explicitly illustrative |

Use reproducible Matplotlib figures in the existing blue/orange palette,
neutral within-model references, line styles and direct labels. No branding or
decorative network of boxes/arrows. Inspect every figure in the complete PDF.
Axes keep the natural similarity and coverage bounds; thresholds are displayed
continuously. Fit variation is not independent chemical replication.

## Reproduction

From the repository root in the canonical `ms2lda-neural` environment:

```bash
python -m scripts.run_cross_model_overlap --output output/benchmarks/cross_model_overlap_new_replay --seal
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 python -m scripts.run_cross_model_overlap --output output/benchmarks/cross_model_overlap_new_replay
python -m scripts.validate_cross_model_overlap --evidence research/cross_model_overlap_20260908/evidence --replay output/benchmarks/cross_model_overlap_new_replay --output output/benchmarks/cross_model_overlap_new_replay_audit.json
python -m scripts.generate_cross_model_overlap_report
pytest -q benchmarks/neural_ms2lda/tests/test_cross_model_overlap.py
```

Use a fresh replay path. The runner defaults to this directory's protocol,
copies it when sealing, and verifies inputs both before and after calculation.
Full replay needs the six original fitted matrices and the preceding chemical
assessment. Report regeneration needs only the committed sealed evidence and
Matplotlib/NumPy; `python -S -m scripts.generate_cross_model_overlap_report
--tables-only` regenerates the text fragments using the standard library alone.
No new inference-speed or DreaMS result is implied by this assessment.

## Evidence schema and validation

`evidence/input_seal.json` records input identities; `protocol.json` fixes
cohort/similarity order, the 201-point threshold grid and example-selection
rules. `complete.json` hashes exactly four scientific payloads:

- `directed_matches.npy`: 133,220 numeric structured rows, without pickles.
  Fit, cohort and similarity codes are positions in the protocol lists;
  topic IDs are zero-based. Scores, first/second targets, reciprocal flags,
  support counts and chemical diagnostics are retained. Undefined chemical
  agreement is NaN, never zero.
- `summary.json`: 180 ordered fit-pair/cohort/similarity summaries, their
  complete 0/1/2+ counterpart curves, 72 source-fit summaries and 24 final
  direction/cohort/similarity groups. Pair medians are averaged over target
  fits within a source fit, then mean and sample SD are reported over the
  three source-fit summaries. Shared fits imply dependence, not nine
  independent cross-model replicates.
- `inventory.json`: the six fit identities, cohort sizes and row counts.
- `examples.json`: deterministic median/upper-decile examples, typed top-20
  words with original probabilities, and separate fragment/loss overlap at
  0.005, 0.01 and 0.02 Da. These are illustrative, not chemically selected.

The completed [validation record](validation.json) verifies byte-identical
full replay of all four payloads. An independent sorted-score implementation
reproduces every coverage curve with zero difference and checks every source
denominator. All 133,220 reciprocal flags agree with reverse matches. Free
nearest-neighbor scores are no lower than earlier Hungarian scores whenever
the forced counterpart remains eligible (30,000 all-topic and 9,834
recurring/evaluable comparisons). This comparison does not discard unmatched
source motifs from the new analysis.

## Findings and interpretation

Mean source-fit summaries of closest-counterpart cosine are 0.454 in both
cross-model directions over all topics. Among recurring, MAG-evaluable
motifs they are 0.420 neural-to-Tomotopy and 0.460 in reverse, versus 0.625
and 0.525 for the corresponding repeat-run benchmarks. The complete curves
show close counterparts in the upper tail, not near-identical inventories.
For those recurring cross-model matches, fragment cosine is about 0.55,
whereas loss cosine is about 0.006--0.007. Equal-channel matching and dropping
the MAG requirement materially change the summaries; no channel-accuracy or
chemical-superiority claim follows from these differences.

The earlier exploratory analysis retained only Hungarian pairs for which
both members were recurring and evaluable **after** matching. Its higher
cross-model median (about 0.638) is not recovery of all eligible source motifs.
The present analysis first fixes each eligible inventory, then finds each
source's best eligible target and retains weak scores. The two summaries
answer different questions. The new definition avoids presenting a favorable
subset of matched pairs as coverage of the entire inventory.

The findings support partial continuity with established MS2LDA spectral
patterns, not independent chemical validation. Chemical-feature agreement
and compound overlap remain modest, and high full-beta cosine can be driven
by one dominant fragment. Similarity is not the probability of correct
chemistry, target reuse is not proof of a chemical split or merge, and an
unmatched motif is not necessarily chemically novel.

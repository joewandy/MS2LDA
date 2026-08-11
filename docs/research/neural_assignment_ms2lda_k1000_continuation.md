# Neural-assignment MS2LDA: exploratory K=1000 continuation

## Why this is a separate protocol

The original bounded run, `neural-assignment-ms2lda-seed42-v1`, is retained
unchanged. It passed both synthetic recovery problems and every K=200
validation criterion except the active-topic screen. At the strict (1/K)
corpus-usage threshold, 67 of 200 topics were active against the prespecified
minimum of 120.

That binary stop concealed a materially better continuous result. The model
was numerically stable, achieved validation NLL 8.292, top-word diversity
0.871, mean NPMI -0.515, and a median 11.03 effective topics per spectrum.
It used 122 topics at or above (0.1/K), and 115 topics accounted for 99% of
validation topic mass. The Tomotopy K=1000 reference retained 363 of 1000
topics under the same strict (1/K) definition. The comparison across K is
diagnostic rather than an equivalence claim, but it makes the K=200 count a
poor hard proxy for final K=1000 viability.

We therefore preserve the original result and declare a post-hoc exploratory
continuation before opening any test matrix. This is not a claim that the
original K=200 gate passed.

## Exact amendment

The committed continuation protocol is
`benchmarks/neural_assignment_ms2lda/protocol_k1000_continuation.json`.
It keeps the original K=200 minimum of 120 in the scorecard, records the raw
failure, and waives only `active_topics` as a blocking K=200 failure. Any
stability, completion-NLL, diversity, NPMI, or mixture failure still stops the
run.

No model, optimizer, initializer, view, training schedule, rescue, runtime
budget, or input changes. In particular, the final gates remain:

- K=1000 validation requires at least 255 active topics, diversity at least
  0.6648, median effective topics from 2.279 to 36.469, and validation NLL at
  most 10.7326.
- Only one collapse-diagnosed rescue is available, from the identical
  initialization.
- Validation selection is frozen before the single test-data touch.
- Chemical evaluation runs only after the selected model passes every
  non-chemical test gate.

## Interpretation

Because the continuation decision used the v1 validation result, its K=1000
validation outcome is exploratory. The untouched test and chemical stages
remain honest evaluations of the validation-selected model, but they do not
retroactively turn the amended screening decision into a prespecified one.

The continuation answers a narrow applied question: does the partial capacity
seen at K=200 scale to a useful K=1000 fully neural motif inventory under the
unchanged final criteria? If not, the preserved result returns the project to
fully neural model design without an automatic annotation redirect.

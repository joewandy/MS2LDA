# Real MSnLib validation-only comparison (seed 42)

No candidate test output was opened. All candidates used the locked split,
train-only V=21233 vocabulary,
train-only 48D SGNS, completion views, leakage-filtered MAG, and SOS
implementation from M1.

## Result

| method | optimized | evaluable | useful | mean SOS | median SOS | completion NLL | frozen gate |
|---|---:|---:|---:|---:|---:|---:|---|
| M1 reference | 884 | 408 | 265 | 0.658079 | 0.648864 | 8.974140 | pass |
| canonical ETM | 609 | 130 | 79 | 0.638796 | 0.631266 | 8.690730 | fail |
| pooled likelihood | 967 | 14 | 11 | 0.751621 | 0.746808 | 8.385787 | fail |
| pooled + MI 0.05 | 799 | 14 | 11 | 0.696540 | 0.645752 | 8.393690 | fail |

All candidates passed the completion-NLL and finite-execution gates, but none
preserved M1's chemical breadth. ETM retained a non-catastrophic inventory (269
topics above 0.0005 usage; corpus effective count 344.6), yet produced only 79
useful validation motifs. The pooled models annotated many topic spectra but
their diffuse document mixtures (median 130.0/124.6 effective topics) almost
never crossed the locked 0.5 association threshold, leaving only 14 evaluable
motifs each. Weak MI did not materially help and was slightly worse on NLL and
annotation coverage.

## ETM channel and collapse diagnostics

ETM fragment mass was materially asymmetric but not uniformly one-sided:
minimum 0.0134, median 0.3420, maximum 0.9973, with 15.5% of topics below 0.1
or above 0.9. No forced-50/50 ETM was run because the predeclared simulation
found no consistent benefit and the real failure was broad chemical retention,
not channel mass alone. ETM did not show catastrophic topic-inventory collapse:
269 topics exceeded 0.0005 usage, corpus effective count was 344.6, and mean
nearest-topic beta cosine was 0.321, although one near-duplicate pair reached
0.9995 and document mixtures were diffuse (median 43.8 effective topics).

## ECRTM feasibility and decision

The canonical full-K/V probe converged in 201 Sinkhorn iterations at residual
0.00475 and used about 1.08 GB peak process memory. At batch 200 it took 4.66
seconds per forward/backward/step, projecting to about 10.6 minutes per epoch
and 7.1 hours for 40 epochs. The labelled 50-step approximation took 1.34
seconds but remained unconverged (residual 1.24). Full ECRTM was not run because
ETM's decisive failure was chemical breadth rather than the topic collapse ECR
is designed to repair; the long comparator was therefore not scientifically
warranted for this first campaign.

## Review surface

`comparison.csv` contains all gates and headline diagnostics. Each model
directory contains its exact protocol, metrics, training history, motif-level
chemical scores, and top words. `provenance.json` records commands, SHAs, asset
identifiers, hardware/software, seeds, and SHA-256 values for uncommitted large
artifacts. `ecrtm_feasibility/` contains the exact and approximate probe evidence.

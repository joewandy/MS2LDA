# Principled sparse-ETM experiment log

Evidence boundary: synthetic truth-known data first; real MSnLib validation
only after a frozen promotion decision. Candidate MSnLib test theta, completion,
MAG, SOS and result artifacts must not be opened, loaded, computed or inspected.

## Predeclared bounded ladder

All synthetic runs reuse 18 planted motifs, paired fragment/loss observations,
train-only vocabulary and 48-dimensional SGNS, intensity pseudo-counts, 800
training and 160 validation spectra, seeds 11/23/37, and 120 ETM epochs. The
first stress condition is K=36; promoted mechanisms are then checked at K=128.

### A. Balanced fixed-SGNS ETM reference

- **Hypothesis:** the reconstructed harness should reproduce the already
  committed balanced-ETM failure pattern closely enough to serve as a paired
  reference: healthy but incomplete beta recovery, strong true-theta recovery,
  and substantially more than the planted 1--3 effective topics per spectrum.
- **Exact model change:** none; fixed-SGNS ETM with independent 0.5 fragment and
  0.5 loss beta normalization, Gaussian posterior, `softmax(z)`, raw
  pseudo-count reconstruction and Gaussian KL.
- **Seed/config:** seed 11 first, K=36, 120 epochs; seeds 23/37 only in bounded
  confirmation.
- **Stopping rule:** stop after 120 epochs or immediately on non-finite loss or
  gradient. If simulator summary statistics or the baseline failure pattern do
  not match the committed protocol, repair the harness before testing a new
  mechanism.

### B1. Balanced ETM with 1.5-entmax theta

- **Hypothesis:** the smoother published sparse transform will lower exact
  support and effective-topic count toward 1--3 while preserving beta recovery,
  inventory breadth and held-out likelihood better than sparsemax.
- **Exact model change:** replace only `softmax(z)` by `entmax15(z)` in stochastic
  training and deterministic inference.
- **Seed/config:** seed 11, K=36, 120 epochs; all other settings paired with A.
- **Stopping rule:** reject on non-finite values, support collapse to one topic
  for most spectra, materially worse true-beta recovery, or severe topic
  starvation. Confirm only if it improves theta sparsity without those failures.

### B2. Balanced ETM with sparsemax theta

- **Hypothesis:** the published Gaussian-sparsemax construction will directly
  remove irrelevant topic mass, but may be too hard and starve topics compared
  with 1.5-entmax.
- **Exact model change:** replace only `softmax(z)` by `sparsemax(z)` in
  stochastic training and deterministic inference.
- **Seed/config:** seed 11, K=36, 120 epochs; all other settings paired with A.
- **Stopping rule:** identical to B1.

### C. Ordinary-theta ETM with distinct-word reconstruction mass

- **Hypothesis:** raw intensity pseudo-count mass overwhelms the Gaussian KL.
  Renormalizing each document's reconstruction weights so their total equals
  its observed nonzero word count will preserve relative intensity evidence
  while restoring a document-length-appropriate reconstruction/KL balance.
- **Exact model change:** keep `softmax(z)` and every model equation unchanged;
  scale each row's raw counts by `nnz(row) / sum_counts(row)` only inside the
  training reconstruction objective. Evaluation still uses the untouched
  pseudo-count completion matrix and reports NLL per pseudo-count token.
- **Seed/config:** seed 11, K=36, 120 epochs; paired optimizer and data.
- **Stopping rule:** reject if beta/theta recovery or inventory deteriorates, or
  if sparsity does not improve materially. Combine with B only if both B and C
  show complementary independent gains.

No sparse prior, gate, separation, NPMI, Sinkhorn or token routing is authorized
by this predeclaration. A prior change is secondary and requires a measured
residual failure after this ladder.

## Seed-11 triage result

All four predeclared CUDA runs completed 120 epochs with finite objectives and
gradients. The reconstructed seed-11 corpus has V=1,833, median 30 physical
peaks, 26 in-vocabulary words and 1,369.5 pseudo-count mass. These closely
match the preserved protocol and historical seed-11 V=1,859 / 29 / 25.5
statistics.

| Formulation | NLL | beta cosine | theta cosine | median effective topics | median exact support | active >0.005 | unique top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| softmax + raw counts | 6.4733 | 0.3114 | 0.4968 | 3.25 | 36 | 8 | 7 |
| 1.5-entmax + raw counts | 6.6167 | 0.2503 | 0.4031 | 1.82 | 3 | 5 | 4 |
| sparsemax + raw counts | 6.6465 | 0.2574 | 0.3822 | 1.90 | 2 | 4 | 4 |
| softmax + distinct-word mass | 6.4152 | 0.4509 | 0.6681 | 28.59 | 36 | 36 | 14 |

The raw-count sparse transforms solve per-document support but worsen beta/theta
recovery and starve the inventory. Distinct-word scaling independently improves
NLL, beta/theta recovery, active topics and unique top-1 topics, but removes the
raw-count pressure that had made softmax relatively concentrated. Thus B and C
show complementary effects, justifying one combination experiment, but neither
is independently promotable.

The new baseline NLL is close to the preserved balanced-ETM seed-11 result
(6.4526), but beta recovery (0.3114 versus 0.4793), theta recovery (0.4968 versus
0.7897) and median effective topics (3.25 versus 5.47) are not close. The
historical exploratory simulator source was deliberately not retained, so the
descriptive protocol cannot recover every generator detail. It would be
scientifically improper to tune the new simulator until old model metrics are
matched. The new screen is therefore an auditable paired reconstruction, and
its absolute metrics will not be presented as a reproduction of the old CSV.

### A-backend integrity diagnostic

- **Hypothesis:** historical results were produced on a different execution
  backend; a CPU replay can determine whether CUDA training randomness explains
  the recovery discrepancy.
- **Exact model change:** none. Reuse the exact same generated seed-11 matrices
  and train-only SGNS, but train the balanced softmax/raw-count baseline on CPU.
- **Seed/config:** seed 11, K=36, 120 epochs, otherwise identical.
- **Stopping rule:** one replay only; do not select a backend by its scientific
  score. If it does not recover the historical regime, retain the discrepancy
  as a simulator-source limitation.

### D. 1.5-entmax theta plus distinct-word reconstruction mass

- **Hypothesis:** distinct-word mass may preserve the broad recovered inventory
  while 1.5-entmax removes irrelevant per-spectrum topic mass. Entmax is chosen
  over sparsemax because it had better NLL and theta recovery in B while still
  reaching median support three.
- **Exact model change:** combine only the already isolated
  `theta=entmax15(z)` transform and `nnz(row)/sum_counts(row)` reconstruction
  weighting. Gaussian posterior/KL, decoder, SGNS, channel balance and optimizer
  remain unchanged.
- **Seed/config:** seed 11, K=36, 120 epochs.
- **Stopping rule:** reject if median support remains far above the planted 1--3,
  or if recovery/inventory remains materially below C. Only a candidate that
  resolves both sides of the measured trade-off advances to seeds 23/37 and
  K=128.

## Backend diagnostic and combination result

The CPU baseline used the identical SGNS hash and produced NLL 6.4130, beta
cosine 0.3400, theta cosine 0.5617 and median 3.34 effective topics. This is a
modest backend shift from CUDA, not recovery of the historical 0.479/0.790
regime. The simulator-source limitation therefore remains, and CUDA is retained
for the paired campaign.

The combined seed-11 formulation completed stably: NLL 6.4104, beta cosine
0.4662, theta cosine 0.7434, all 36 topics active above 0.0005, 13 unique top-1
topics and no catastrophic strict duplicate component. Median effective topics
and exact support were both 1, with 70% support <=3. Support is bimodal rather
than uniformly sparse: mean support 9.68, 75th percentile 18.25 and 95th
percentile 36. This candidate resolves much of the independent B/C trade-off,
but the high-support tail must be stress-tested rather than hidden by the
median.

### D-confirm. Frozen multi-seed confirmation

- **Hypothesis:** the entmax plus distinct-word result is a mechanism effect,
  not a seed-11 initialization accident.
- **Exact model change:** none beyond frozen D. Pair it with the raw-count
  softmax reference A and the independently useful scaling control C.
- **Seed/config:** seeds 23 and 37, K=36, 120 epochs, CUDA. No parameter changes.
- **Stopping rule:** reject real promotion if D loses the recovery/inventory
  improvement, collapses materially used topics, becomes non-finite, or fails
  to retain substantially sparser theta than C on either seed.

### K128. More-overcomplete stress

- **Hypothesis:** a principled sparse formulation should retain its benefit when
  fitted K rises from 2x to about 7x the planted motif count.
- **Exact model change:** topic count only. Compare A, C and frozen D.
- **Seed/config:** seed 11, 18 true motifs, fitted K=128, 120 epochs, CUDA; same
  800/160 corpus and train-only SGNS.
- **Stopping rule:** do not run K=256/1000 automatically. Reject promotion if D
  shows catastrophic inventory duplication/starvation, loses recovery relative
  to controls, or its per-document support scales approximately with K.

## Multi-seed and high-K decision

At K=36, frozen D completed all seeds stably. Three-seed means were NLL 6.3930,
beta cosine 0.4530, theta cosine 0.7304, median effective topics 1.0, mean
effective topics 3.68, 36/36 topics active above 0.0005, 11.7 unique top-1
topics and mean nearest-beta cosine 0.8936. It improved NLL and beta/theta
recovery relative to both paired controls on every seed. Exact support remained
bimodal (median 1; mean 9.93; mean fraction support <=3 of 0.681).

At K=128, D retained better beta recovery (0.4357) and theta recovery (0.6932)
than raw softmax (0.3696/0.5782) and scaled softmax (0.4202/0.4876). Its median
effective topics were 3.02 versus 3.67 and 127.29, respectively. All 128 topics
remained active above 0.0005, with 12 unique top-1 topics and no component at
cosine >=0.999 large enough to be catastrophic. However, exact median support
rose to 52, the 95th percentile was 128 and mean nearest-beta cosine rose to
0.9760. Thus concentration survives high K better than exact sparsity or beta
distinctness.

The candidate meets the minimum promotion rule—consistent recovery gains,
substantially less diffuseness than the scaling control, broad usage, finite
training and recognizable ETM equations—but carries an explicit high-K warning.
It is promoted as the sole real validation candidate. No sparse-prior experiment
is authorized before seeing whether this warning materializes at real K=1000.

### Real MSnLib validation: promoted D only

- **Hypothesis:** training-time 1.5-entmax plus document-distinct reconstruction
  mass will give K=1000 ETM mixtures near the short-spectrum scale while
  preserving the balanced ETM's broad global topic inventory and chemistry.
- **Exact model change:** relative to balanced fixed-SGNS ETM, replace
  `softmax(z)` by `entmax15(z)` and multiply each training row's raw-count
  reconstruction weights by `nnz(row)/sum_counts(row)`. Keep the Gaussian
  posterior and standard-normal KL, fixed train-only 48D SGNS, 50/50
  fragment/loss beta, encoder/decoder, optimizer and evaluation unchanged.
- **Data/evidence:** link only frozen train and validation matrices/records,
  vocabulary, train-only SGNS and the existing leakage-filtered MAG index into a
  new writable run view. Do not expose or access candidate-test artifacts.
- **Config:** prepared seed 42; model seed 7043; K=1000; hidden=800; Adam
  lr=0.005, weight decay 1.2e-6; batch size 256; 120 epochs; CUDA; six CPU
  threads; checkpoint every five epochs.
- **Stopping rule:** fail immediately on non-finite objective/gradient, OOM or
  integrity mismatch. After 120 epochs run unchanged completion, diagnostics,
  MAG and SOS on validation only. Apply every frozen gate as written; do not
  rescue, recalibrate or tune this run.

A disposable one-epoch K=1000 smoke completed training in 7.28 seconds with
finite objective/gradients and wrote the expected validation-only artifacts.
Its initial reporting failure was a wiring error: inventory thresholds reside
in the established comparison runner rather than the prepared protocol. The
exact frozen diagnostic mapping was reused and the resumed smoke then completed;
no scientific setting changed.

## K=1000 numerical simplex correction

The first full real attempt completed all 120 epochs with finite losses and
gradients, then failed closed before metrics were saved because validation
observed entmax rows did not meet the explicit `atol=2e-6` sum-to-one contract.
The row-sum range was 0.999521534--1.00031805, a maximum deviation of
0.000478466. The same measured implementation error grows with K: maximum
deviation 0.00000376 at synthetic K=36 and 0.0000250 at K=128.

- **Diagnosis:** this is finite-precision accumulation in the float32 entmax
  simplex kernel, not NaN/Inf, negative probability, divergence or an OOM.
- **Exact numerical change:** divide every transform output row by its computed
  positive total. This is not a temperature or scientific hyperparameter; it
  preserves exact zeros, topic ranks and relative nonzero weights while
  enforcing the probability-simplex definition of sparsemax/entmax.
- **Verification:** add a K=1000 simplex/exact-zero test and repeat frozen D at
  seeds 11/23/37, K=36, plus seed 11 K=128 in a fresh numerical-check root.
- **Real stopping rule:** do not resume the old checkpoint under changed
  numerics. If the synthetic direction is preserved, rerun the same K=1000
  config from initialization in a fresh validation-only output. Preserve the
  failed pre-correction checkpoint and logs as provenance.

The normalized-transform recheck preserved the decision. K=36 three-seed means
were NLL 6.3863, beta cosine 0.4546, theta cosine 0.7422, median effective/exact
support 1/1, 36/36 topics active above 0.0005 and 11.7 unique top-1 topics.
Relative to the pre-correction run, changes were small and directionally mixed,
while the recovery advantage over both controls remained. At K=128, beta/theta
recovery remained 0.4374/0.6896, all 128 topics remained active, median
effective topics were 3.85, and exact median support improved from 52 to 15.5;
the high-support tail and high redundancy remained. The real run is therefore
restarted unchanged under `real_validation_simplex_normalized`; the original
completed-training/reporting-failed run remains preserved under
`real_validation`.

## Corrected real validation result

The fresh normalized-transform run completed all 120 epochs without NaN, Inf,
OOM or a failed gradient check. It passed the strict non-negative simplex
contract and deterministic inference checks. Training took 791.31 seconds on
CUDA; validation observed/full inference measured 32,677/31,242 spectra per
second. PyTorch peak allocated/reserved CUDA memory was 0.811/1.028 GB, process
peak memory was 2.839 GB, and minimum sampled system-available memory was
12.875 GB.

Completion NLL was 9.577829 with OOV fraction 0.031356. The sparse transform
changed the median dramatically but produced a severe two-regime distribution:

- median effective topics 1.803 and median exact support 2;
- mean effective topics 140.605 and mean exact support 295.257;
- exact-support p75/p95/p99 of 789/994/997;
- 55.4% of spectra at support <=3;
- median maximum theta 0.772 and 63.4% at maximum theta >=0.5.

The global inventory did not remain broad: 20 unique top-1 topics, 98 active
above 0.0005, 39 at or above `1/K`, corpus-effective count 56.98 and maximum
mean use 0.1765. Mean nearest-beta cosine was 0.9764 and maximum beta cosine
0.9973. The largest 0.95 duplicate component contained 941 topics; no pair
reached the frozen strict 0.999 threshold, so the formal catastrophic-duplicate
gate still passed.

### Unchanged validation chemistry

- **Command/evidence:** the promoted validation arrays were evaluated through
  the unchanged `chemical()` validation path, the frozen membership threshold,
  existing leakage-filtered MAG index and SOS implementation. Candidate test
  chemistry was neither loaded nor computed.
- **Result:** 993 optimized, 7 evaluable, 6 useful, mean SOS 0.663853, median
  SOS 0.686250. SOS bands were 0 high, 6 intermediate and 1 low; membership
  associated 2,464 spectra and 2,436 molecules. The held-out-compound leakage
  exclusion audit passed.
- **Frozen gates:** optimized pass; evaluable fail; useful fail; mean SOS pass;
  completion NLL fail; finite/stable pass; no catastrophic strict duplicate
  pass. Overall result: **fail**.

The mean SOS cannot be interpreted as broad chemical success because it is
computed over only seven eligible motifs. The exact failure is a hard
topic-starvation regime coexisting with an extremely diffuse document tail,
near-duplicate beta geometry, almost no chemically evaluable inventory, and
degraded completion likelihood. This is not a remaining post-hoc calibration
problem.

## Final campaign decision

The evidence rejects the proposition that replacing dense softmax is sufficient
to rescue ETM for K=1000 short MS/MS spectra. Pseudo-count scaling matters for
recovery, and entmax can make the median document exactly sparse, but the
combination does not maintain a broad, chemically useful topic inventory or
uniformly sparse support at real K.

No sparse prior was tested, so this campaign does not claim it is ineffective.
It would be a materially larger model-family change, and the current bounded
evidence does not justify making it the next campaign. Stop ETM development at
this checkpoint. The next authorized research campaign should be M1 multiseed
stability; it was not started here.

## Final quality-control audit

Focused sparse-ETM/simulator/isolation tests passed 14/14, the complete neural
suite passed 75/75, and the exact repository-CI production command passed
87/87 with two existing empty-document NumPy warnings. Black, Ruff, JSON/CSV
parsing, `git diff --check`, finite-gradient, non-negativity, simplex,
exact-zero, deterministic-inference, validation/test-isolation and binary-size
checks passed.

An over-broad initial `pytest -q tests` invocation included four legacy files
that `.github/workflows/neural-ms2lda.yml` explicitly excludes. It produced 134
passes and 33 failures in untouched callback/corpus/integration/download
interfaces, and is retained here as a disclosed non-CI diagnostic rather than
misreported as a pass. Its download tests created two ignored 5.7 GB model
directories inside the checkout. Both paths were verified untracked and newly
created by that invocation, then removed; the campaign's preserved shared
Spec2Vec and MAG assets were not modified.

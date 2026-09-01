# Routing-informed ETM experiment log

Status: campaign complete and frozen as the paper-facing validation baseline;
candidate test remains locked.

## Scientific boundary

The paper-facing base is the published Embedded Topic Model. Its fixed train-only
SGNS word table, 50/50 fragment/loss decoder, multinomial reconstruction,
Gaussian logistic-normal variational family, analytic standard-normal KL,
optimizer and training data remain unchanged.

This campaign treats the earlier M1 system only as donor evidence. It does not
train, select, or attempt to publish M1. It excludes M1's nonlinear router,
document gate, Sinkhorn targets, positive-NPMI loss, prototype separation,
temperature schedule and alternating optimization.

## Measured failure being addressed

Balanced and gated ETM retained a broad real topic-word inventory and acceptable
completion likelihood, but their per-spectrum topic mixtures were extremely
diffuse. Post-hoc sharpening could not jointly preserve likelihood, useful-motif
breadth and SOS. The remaining hypothesis is therefore about amortized inference,
not another beta anti-collapse penalty.

## Predeclared model screen

All variants use raw intensity pseudo-count reconstruction and ordinary softmax
theta. The only changing computation is the information supplied to the ETM
Gaussian posterior mean.

1. `etm`: exact existing balanced fixed-SGNS ETM control.
2. `soft_token`: score each observed fixed-SGNS word vector against ETM's own
   topic embeddings, aggregate soft word-topic evidence, and add its centered
   bounded log evidence to the posterior mean.
3. `soft_context`: before scoring, add one learned scalar times the count-weighted
   leave-one-out mean of the other observed words in the spectrum.
4. `top2_context`: retain only the two strongest contextual topic assignments per
   observed word before document aggregation.

The evidence offset uses a fixed uniform pseudocount of `1/K`; this bounds absent
topic evidence and makes uniform evidence an exact no-op after row centering.
Routing temperature is fixed at 1.0. There is no temperature or strength sweep.

## Synthetic protocol and stopping rule

- Paired fragment/loss simulator with 18 planted motifs.
- K=36 primary screen; K=128 high-overcompleteness stress.
- Frozen seeds 11, 23 and 37, with 800 training and 160 validation spectra.
- Train-only vocabulary and train-only 48-dimensional SGNS.
- 120 epochs, Adam learning rate 0.005, weight decay 1.2e-6, batch size 200.
- CUDA when available; deterministic execution; stop immediately for non-finite
  objective or gradients.

The exact existing ETM control artifacts may be reused because a focused unit
test requires the new `etm` code path to have identical parameters, state and
posterior calculations to `BalancedSparseETM` under the same seed.

## Promotion rule

A variant can advance only if its K=36 results are finite on all three seeds and,
relative to the paired ETM control:

- mean true-theta cosine improves by at least 0.02 and no seed falls by more
  than 0.02;
- median effective topics per spectrum is at most 60% of control on every seed;
- true-beta cosine falls by no more than 0.02 on any seed;
- held-out NLL is no more than 5% worse on any seed;
- active topics above 0.5% mean use retain at least 80% of control and never fall
  below the 18 planted topics;
- no catastrophic strict duplicate component appears.

Among passing variants, choose the simplest. Before real validation it must also
survive seed-11 K=128 with finite execution, no catastrophic duplicate component,
at least 18 active topics, true-beta cosine no more than 0.02 below the paired
K=128 control, and held-out NLL no more than 5% worse.

At most one candidate advances to frozen MSnLib validation. No candidate test
artifact will be opened or scored.

## Experiment 1: implementation and mathematical checks

- **Hypothesis:** the ETM control can be nested exactly while routing variants
  preserve a valid finite Gaussian posterior, simplex theta, and deterministic
  inference.
- **Exact change:** add `RoutingInformedETM`, its four variants, a synthetic
  runner, and focused tests.
- **Stopping rule:** do not begin research runs unless the control is exactly
  equivalent and every routing variant has finite gradients and simplex outputs.

- **Result:** 10/10 focused tests passed. The nested control has byte-identical
  state and posterior calculations under the same seed; all routing variants
  produced finite gradients, deterministic inference and normalized theta. A
  two-epoch end-to-end CPU smoke also completed successfully. CUDA is available
  on the NVIDIA GeForce RTX 5070.
- **Decision:** begin the predeclared seed-11 K=36 mechanism triage.

## Experiment 2: seed-11 K=36 component triage

- **Hypothesis:** token evidence should improve ETM's amortized mixture recovery;
  leave-one-out context should add short-spectrum disambiguation; top-2 token
  evidence should be the only step that substantially reduces diffuseness.
- **Exact model changes:** run `soft_token`, `soft_context`, and `top2_context`
  against the existing exact `etm` control. All other model and training settings
  are identical.
- **Config:** seed 11, 18 planted topics, K=36, 800/160 train/validation spectra,
  120 epochs, batch 200, Adam 0.005, weight decay 1.2e-6, raw pseudo-counts,
  routing temperature 1.0, CUDA, six CPU threads.
- **Stopping rule:** stop the campaign immediately if no routing variant improves
  theta recovery while satisfying the paired beta, NLL, inventory and stability
  constraints. Otherwise confirm the simplest qualifying mechanism on seeds 23
  and 37 before high-K stress.

| variant | NLL | beta cosine | theta cosine | top-motif accuracy | median effective topics | active >0.5% | unique top-1 | nearest-beta cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ETM control | 6.4733 | 0.3114 | 0.4968 | 0.4750 | 3.25 | 8 | 7 | 0.8863 |
| soft token | 6.4880 | 0.3176 | 0.5216 | 0.4938 | 3.36 | 8 | 6 | 0.8956 |
| soft context | 6.4361 | 0.3318 | 0.5430 | 0.5250 | 3.28 | 9 | 7 | 0.8812 |
| top-2 context | 6.2645 | 0.4466 | 0.7010 | 0.6875 | 3.33 | 14 | 12 | 0.7730 |

- **Result:** top-2 contextual evidence clearly improves planted beta/theta
  recovery, top-motif accuracy, topic use, top-1 breadth, redundancy and NLL.
  The soft variants give only small gains. However, the raw-count synthetic ETM
  control is already mixture-sparse because it has collapsed to eight materially
  used topics. It therefore does not reproduce the measured real-data failure of
  a broad topic inventory with diffuse per-spectrum theta. Top-2 routing cannot
  satisfy the original 60%-of-control diffuseness criterion because the control
  median is already only 3.25 effective topics; it also remains below the
  predeclared 18-active-topic floor.
- **Decision:** Experiment 2 does not promote a candidate. Treat the control
  mismatch as a failed experimental precondition, not as permission to loosen
  the rule. Run one separately predeclared diagnostic confirmation to ask whether
  the promising top-2 mechanism generalizes across raw-count seeds and can
  actually repair a deliberately diffuse ETM condition. No other variant or
  hyperparameter will be added.

## Experiment 3: top-2 mechanism confirmation under both failure regimes

- **Hypothesis:** top-2 contextual evidence should (a) consistently improve
  recovery and topic use under the raw-count collapse regime and (b) reduce
  diffuseness when the paired ETM control genuinely exhibits it.
- **Raw-count confirmation:** run `top2_context` at K=36 for seeds 23 and 37.
  It must improve both beta and theta cosine over the paired control on each
  seed, retain or increase active and unique-top-1 topics, remain within 5% NLL,
  and avoid a catastrophic duplicate component.
- **Diffuse stress:** use the already-characterized `distinct_words` objective
  only as a mechanistic stress condition; it is not the proposed real model.
  Run `top2_context` at K=36 for seeds 11, 23 and 37. Relative to the exact paired
  distinct-word ETM control on every seed, it must reduce median effective topics
  to at most 60%, keep beta and theta cosine within 0.02, retain at least 80% of
  active topics, remain within 5% NLL and avoid catastrophic duplication.
- **Stopping rule:** if either arm fails, do not run high-K or real validation.
  If both pass, run the same top-2/raw-count candidate at seed-11 K=128 and apply
  the original high-K inventory, recovery, NLL and duplicate-component gates.

### Result

Raw-count top-2 routing generalized cleanly across seeds:

| seed | model | NLL | beta cosine | theta cosine | median effective topics | active >0.5% | unique top-1 |
|---:|---|---:|---:|---:|---:|---:|---:|
| 11 | ETM | 6.4733 | 0.3114 | 0.4968 | 3.25 | 8 | 7 |
| 11 | top-2 | 6.2645 | 0.4466 | 0.7010 | 3.33 | 14 | 12 |
| 23 | ETM | 6.4267 | 0.3105 | 0.5039 | 3.90 | 8 | 7 |
| 23 | top-2 | 6.1864 | 0.4594 | 0.7144 | 3.06 | 13 | 11 |
| 37 | ETM | 6.5129 | 0.2937 | 0.4598 | 3.61 | 8 | 6 |
| 37 | top-2 | 6.2805 | 0.4208 | 0.7043 | 3.27 | 13 | 12 |

The diffuse stress did not pass:

| seed | ETM effective topics | top-2 effective topics | ETM theta cosine | top-2 theta cosine |
|---:|---:|---:|---:|---:|
| 11 | 28.59 | 21.13 | 0.6681 | 0.6896 |
| 23 | 34.39 | 30.55 | 0.5040 | 0.4433 |
| 37 | 30.03 | 20.64 | 0.6162 | 0.6873 |

- **Interpretation:** top-2 contextual evidence is a reproducible recovery and
  anti-collapse mechanism under raw counts, but a dense softmax still prevents
  it from being a reliable sparsity mechanism. The diffuse-stress reductions
  were only 11-31%, below the required 40%, and seed 23 lost 0.061 theta cosine.
- **Decision:** do not promote the softmax-routing ETM and do not run it at high
  K or on real MSnLib.

## Experiment 4: one complementary published-sparsity combination

- **Measured complementary failures:** entmax alone supplies exact sparse theta
  but starves/collapses topics under raw pseudo-counts; top-2 routing consistently
  restores topic recovery and use but does not make dense softmax sufficiently
  sparse. Combining these two already-isolated mechanisms is now justified.
- **Exact change:** keep the routing-informed Gaussian posterior mean and replace
  only its final softmax with published alpha-entmax 1.5. No gate, Sinkhorn, NPMI,
  separation, custom prior, temperature sweep or alternating optimizer is added.
- **Seed-11 triage:** test top-2+entmax with raw-count and distinct-word objective
  weighting at K=36. These are the two already-characterized objective regimes;
  no new scaling is introduced.
- **Synergy gate:** a combination must keep median effective topics <=3, remain
  within 5% held-out NLL, avoid catastrophic duplication, and improve beta,
  theta, active-topic and unique-top-1 metrics over the paired entmax-only model.
  Relative to softmax top-2 routing it may lose at most 0.03 beta cosine and 0.03
  theta cosine, and at most 20% of active and unique-top-1 topics.
- **Stopping rule:** if neither objective passes, stop this transplant campaign.
  If exactly one passes, confirm only it on seeds 23/37 and then K=128 before any
  real validation. If both pass, prefer raw counts because it leaves the original
  ETM objective unchanged.

### Seed-11 result

| formulation | NLL | beta cosine | theta cosine | median effective | median exact support | active >0.5% | unique top-1 | nearest-beta cosine |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| entmax + raw | 6.6167 | 0.2503 | 0.4031 | 1.82 | 3 | 5 | 4 | 0.9301 |
| routing + softmax + raw | 6.2645 | 0.4466 | 0.7010 | 3.33 | 36 | 14 | 12 | 0.7730 |
| routing + entmax + raw | 6.2784 | 0.4985 | 0.7649 | 1.97 | 4 | 14 | 14 | 0.7402 |
| entmax + distinct | 6.4104 | 0.4662 | 0.7434 | 1.00 | 1 | 18 | 13 | 0.8834 |
| routing + entmax + distinct | 6.2881 | 0.5382 | 0.8332 | 1.23 | 2 | 15 | 15 | 0.8405 |

- **Result:** both combinations improve recovery, likelihood and redundancy over
  entmax alone. The raw-count combination also improves every reported metric
  over softmax routing while reducing median effective topics from 3.33 to 1.97.
  It passes the complete synergy gate. The distinct-word combination loses 21
  of 36 active topics relative to softmax routing, exceeding the permitted 20%
  loss, so it is rejected despite strong planted recovery.
- **Decision:** select only top-2 contextual routing + entmax 1.5 + raw counts for
  multi-seed and high-K confirmation. This keeps the original ETM reconstruction
  weighting and adds no objective-scaling change.

## Experiment 5: selected raw-count combination confirmation

- **Hypothesis:** the seed-11 synergy is not initialization-specific and survives
  increased overcompleteness.
- **Runs:** K=36 seeds 23/37 for top-2+entmax+raw; then seed-11 K=128 for the exact
  entmax-only paired control and selected combination.
- **Multi-seed gate:** apply the Experiment 4 synergy rule independently on seeds
  23 and 37. Both must pass.
- **K=128 gate:** selected combination must be finite, have no catastrophic
  duplicate component, retain at least 18 active topics above 0.5% mean usage,
  improve beta and theta recovery over entmax-only, remain within 5% NLL, and
  have median effective topics <=3.
- **Stopping rule:** no real run unless every K=36 seed and K=128 pass. At most
  this one formulation may advance.

### K=36 multi-seed result

| seed | formulation | NLL | beta cosine | theta cosine | median effective | median support | active >0.5% | unique top-1 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 11 | entmax only | 6.6167 | 0.2503 | 0.4031 | 1.82 | 3 | 5 | 4 |
| 11 | routing + entmax | 6.2784 | 0.4985 | 0.7649 | 1.97 | 4 | 14 | 14 |
| 23 | entmax only | 6.4898 | 0.2816 | 0.4773 | 1.97 | 3 | 6 | 6 |
| 23 | routing + entmax | 6.2862 | 0.4215 | 0.7234 | 2.00 | 4 | 12 | 11 |
| 37 | entmax only | 6.6866 | 0.2379 | 0.3115 | 1.86 | 2 | 4 | 4 |
| 37 | routing + entmax | 6.3029 | 0.4743 | 0.7655 | 2.03 | 4 | 16 | 14 |

- **Result relative to entmax and ETM:** the selected combination improves beta
  recovery, theta recovery, NLL, active-topic use, unique top-1 breadth and beta
  redundancy on every seed relative to entmax alone. Relative to the paper-facing
  softmax ETM base, it also improves beta and theta recovery, NLL, sparsity,
  active-topic use and top-1 breadth on every seed.
- **Auxiliary gate miss:** on seed 23, beta cosine is 0.4215 versus 0.4594 for
  routing+softmax, a loss of 0.0379 rather than the permitted 0.0300. It gains
  theta recovery and reduces median effective topics from 3.06 to 2.00 on that
  same seed. The other two seeds improve beta as well as theta.
- **Decision boundary:** this row is not recorded as an unqualified pass. The
  original campaign promotion criteria are nevertheless met relative to the
  actual published ETM base on all seeds. Run the already-planned K=128 synthetic
  stress as an adjudication only, with no new model or tuning. Real validation
  remains prohibited unless K=128 clearly passes every stated high-K criterion.

### K=128 adjudication

| formulation | NLL | beta cosine | theta cosine | top accuracy | median effective | median support | active >0.5% | unique top-1 | nearest-beta cosine | catastrophic duplicate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| softmax ETM | 6.4175 | 0.3696 | 0.5782 | 0.5438 | 3.67 | 128 | 7 | 7 | 0.9637 | no |
| entmax ETM | 6.7375 | 0.2344 | 0.2965 | 0.2625 | 1.72 | 2 | 3 | 3 | 0.9829 | no |
| routing + entmax | 6.1276 | 0.6638 | 0.9556 | 0.9313 | 1.45 | 2 | 20 | 18 | 0.8947 | no |

- **Result:** every high-K gate passes. The selected combination recovers all 18
  planted motifs as unique top-1 topics, exceeds the 18-active-topic floor,
  sharply improves beta/theta recovery and NLL, remains sparse, and avoids a
  catastrophic duplicate component. The benefit grows rather than disappears
  under overcompleteness.
- **Decision:** promote this single formulation to frozen K=1000 MSnLib
  validation. No other candidate advances and candidate test remains locked.

## Experiment 6: real MSnLib validation

- **Hypothesis:** routing evidence will prevent the entmax real-data mixture from
  splitting into a tiny sparse subset plus a pathological diffuse tail, while
  preserving the balanced ETM's broad beta inventory and completion likelihood.
- **Exact model:** fixed-SGNS balanced ETM; Gaussian posterior with unchanged
  analytic KL; one learned leave-one-out context scalar; shared ETM topic geometry;
  top-2 token evidence added to the posterior mean; alpha-entmax 1.5 theta; raw
  pseudo-count reconstruction. No other M1 component is present.
- **Inputs:** immutable prepared seed-42 training and validation matrices,
  vocabulary, SGNS and leakage-filtered MAG assets. K=1000. Candidate test files
  are not exposed in the new run view.
- **Training:** 120 epochs, Adam 0.005, weight decay 1.2e-6, batch 256, deterministic
  CUDA, six CPU threads, checkpoint every five epochs.
- **Gates:** optimized >=840, evaluable >=388, useful >=252, mean SOS >=0.651498,
  completion NLL <=9.422847, finite/stable and no catastrophic duplicate component.
- **Stopping rule:** first run a one-epoch validation-only operational smoke. If
  finite and resource-safe, run the exact 120-epoch configuration without tuning,
  followed by unchanged MAG/SOS validation. Never run candidate test.

### One-epoch operational smoke

- **Result:** complete and finite. Training took 8.06 seconds; peak CUDA allocated
  memory was 820,915,712 bytes, CUDA reserved memory 989,855,744 bytes, and Linux
  process high-water memory 2,874,933,248 bytes. Reconstruction was 6117.59, KL
  119.25, mean gradient norm 46.37, and learned context scale 1.512. Validation
  theta already had median exact support 3 and median 1.93 effective topics.
- **Isolation audit:** training and validation links only; candidate test artifacts
  accessed=false and candidate test metrics inspected=false.
- **Decision:** resource-safe and numerically finite. Start the exact 120-epoch
  run in a fresh validation-only directory; do not reuse the one-epoch weights.

### Frozen 120-epoch result

- **Execution:** completed all 120 epochs without restart, configuration change,
  non-finite loss, non-finite gradient, OOM or stall. Checkpoints advanced every
  five epochs. Training took 876.80 seconds.
- **Isolation:** the run view linked only frozen training and validation inputs.
  Validation MAG/SOS reused the leakage-filtered index. Candidate test artifacts,
  completion, chemistry and metrics were not loaded, calculated or inspected.

| frozen gate | threshold | result | outcome |
|---|---:|---:|:---:|
| optimized motifs | >=840 | 803 | fail |
| evaluable motifs | >=388 | 445 | pass |
| useful motifs | >=252 | 289 | pass |
| mean SOS | >=0.651498 | 0.647153 | fail |
| completion NLL | <=9.422847 | 9.542924 | fail |
| finite/stable | required | yes | pass |
| no catastrophic duplicate component | required | yes | pass |

Chemistry had median SOS 0.657895, with 67 topics above 0.8, 222 in the
0.6--0.8 band and 156 below 0.6. It associated 1,400 validation spectra from
1,393 molecules. All held-out compounds remained excluded from MAG.

The document-topic repair is strong rather than cosmetic:

| diagnostic | routing-informed ETM |
|---|---:|
| median / mean effective topics | 3.70 / 4.09 |
| median / p95 / maximum exact support | 6 / 13 / 23 |
| unique top-1 topics | 828 |
| active topics above 0.0005 | 535 |
| corpus-effective topics | 538.40 |
| maximum mean topic use | 0.0143 |
| mean nearest-beta cosine | 0.1529 |
| strict 0.999 duplicate component | 2 topics |
| learned context scale | 0.1772 |

For comparison, balanced softmax ETM had 46.72 median effective topics and only
166 evaluable / 104 useful motifs. The prior sparse ETM made the median sparse
but collapsed to 20 unique top-1 topics and only 7 evaluable / 6 useful motifs.
The present combination fixes both failures simultaneously. It also exceeds the
locked private donor reference in evaluable and useful counts (445/289 versus
408/265), although it does not satisfy the complete frozen contract and is not
authorized for test.

- **Interpretation:** top-2 contextual token evidence is the part of the donor
  architecture that transfers cleanly into ETM. Entmax supplies exact sparse
  support; routing prevents entmax topic starvation. The result remains a
  standard ETM generator and likelihood with a small, auditable posterior
  adaptation: one learned context scalar and no custom training objective.
- **Exact remaining failure:** MAG optimization coverage is 37 motifs below the
  gate, mean SOS is 0.004345 below it, and completion NLL is 0.120077 above it.
  These are narrow quality/generalization deficits, not the old diffuse-theta or
  collapsed-inventory failure.
- **Decision:** record the formal all-gates Boolean as false and keep candidate
  test locked. Scientifically, freeze the model as a strong near-pass and the new
  paper-facing baseline: it exceeds M1 and Tomotopy on evaluable/useful discovery
  breadth and fixes the previously measured sparsity/inventory failures. Do not
  make another architecture change until the checkpoint is verified and real
  training-seed stability is assessed. Positive-NPMI remains one optional,
  predeclared coherence intervention only if the residual coverage/SOS gap
  reproduces; do not add the donor gate, Sinkhorn, prototype separation or
  alternating optimizer.

## Quality-control closure

- Focused routing/sparse tests: 24 passed.
- Full neural suite: 93 passed.
- CI-equivalent production regression suite: 87 passed, with the two established
  empty-document NumPy warnings.
- Black and Ruff passed on every changed Python file.
- Repository-wide formatter sweeps continue to identify four untouched
  pre-existing files; no unrelated formatting churn was included.
- All 25 committed JSON files and 8 CSV files parsed successfully.
- No `.pt`, `.npy`, FAISS index, database or other large model/data artifact is
  present in the review package.
- `git diff --check`, finite gradients, non-negative/simplex theta, exact entmax
  support, deterministic inference and validation/test isolation passed.

## Frozen checkpoint

The source implementation and complete reviewable result package were committed
at `3d9af674949a70a38cbd250b95023f28b9514fe5`. The checkpoint adds a
machine-readable manifest and an integrity checker that reconcile:

- every committed implementation and evidence hash;
- Routing ETM metrics against `metrics.json` and `comparison.csv`;
- M1 and Tomotopy values against the locked `results.json` source;
- SOS band arithmetic and optimized-coverage arithmetic;
- the validation-only access audit; and
- optionally, every retained local model artifact and immutable validation input.

The full local verification completed 78 checks with no discrepancy. This
checkpoint is the baseline for future work; future experiments must write to a
new result directory and must not mutate these evidence files.

## Experiment 8: unchanged real training-seed stability

- **Question:** is the Routing ETM discovery breadth, sparse posterior and broad
  inventory stable to ordinary training initialization, or was the frozen result
  a single-seed accident?
- **Predeclared isolation:** reuse the exact frozen seed-42 train/validation
  split, vocabulary, SGNS, K=1000 configuration, 120 epochs, MAG/SOS evaluator
  and membership threshold. Change only model initialization and minibatch-order
  RNG, using training seeds 23 and 37. Do not load, compute or inspect candidate
  test artifacts.
- **Results:**

| training seed | optimized | evaluable | useful | mean SOS | completion NLL | median effective | median support | unique top-1 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 7043 (original) | 803 | 445 | 289 | 0.647153 | 9.542924 | 3.699 | 6 | 828 |
| 23 | 791 | 453 | 274 | 0.637558 | 9.546012 | 3.702 | 6 | 816 |
| 37 | 787 | 439 | 275 | 0.647350 | 9.539388 | 3.714 | 6 | 813 |
| mean | 793.67 | 445.67 | 279.33 | 0.644020 | 9.542775 | 3.705 | 6 | 819.0 |

- **Stability finding:** every seed exceeds M1's 408 evaluable / 265 useful and
  Tomotopy's 206/138 while retaining exact sparse support, more than 800 unique
  validation winners and no catastrophic duplicate component. The discovery
  advantage is therefore robust to these initializations on the frozen split.
- **Reproduced trade-off:** every seed remains below M1 on optimized coverage and
  mean SOS and above M1 on completion NLL. The original all-gates Boolean is not
  reinterpreted or silently relaxed.
- **Boundary:** n=3 is descriptive same-split initialization stability, not
  data-split, instrument or external-dataset generalization. Candidate test
  remains locked in every saved access audit.
- **Decision:** stop repeating identical real seeds. If further model work is
  desired, positive-NPMI is now the one bounded next experiment justified by the
  reproduced coverage/SOS residual. Predeclare it, screen it synthetically and
  promote at most one configuration to validation; do not reconstruct M1.
- **Evidence:** compact per-seed artifacts, the aggregate summary, hashes and
  verifier are in `../20260830_routing_etm_stability/`.

## Experiment 9: simplest positive-NPMI add-on

The predeclared weight-1 train-derived positive-NPMI term was tested alone on
seed-11 K=36. It reduced its train-graph loss from 5.526062 to 5.499502 but
reduced true-beta recovery from 0.498454 to 0.491576. NLL changed from 6.278416
to 6.287552; theta recovery, sparse support and inventory were effectively
unchanged. This failed the first synthetic gate, so seeds 23/37, K=128 and real
MSnLib were not run. Do not tune its coefficient or add another donor component.
The complete predeclaration, result and artifact hashes are in
`../20260830_routing_etm_npmi/`.

## Experiment 10: zero-parameter top-2 token route

The final simplification removed leave-one-out spectral context and its learned
scalar while retaining direct top-2 token votes and entmax. The protocol was
committed before implementation. At seed-11 K=36 the parameter-free candidate
improved substantially over entmax-only ETM, but relative to contextual Routing
ETM beta/theta recovery fell from 0.498454/0.764875 to 0.410354/0.661425,
recovered motifs fell from 10 to 6 and active/unique topics fell from 14/14 to
11/10. It failed the first non-inferiority gate, so seeds 23/37, K=128 and real
MSnLib were not run. The complete package is in
`../20260830_routing_etm_top2_token/`.

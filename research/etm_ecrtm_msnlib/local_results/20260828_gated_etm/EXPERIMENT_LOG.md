# Real-MSnLib gated ETM experiment log

Evidence boundary: validation only. Candidate test theta, completion matrices,
chemistry, MAG/SOS, and result artifacts must not be opened, loaded, computed,
or summarized during this campaign.

## 1. Frozen balanced ETM plus detached shared-geometry gate

- **Hypothesis:** training balanced fixed-SGNS ETM with detached evidence from
  the same SGNS/topic geometry will reduce diffuse document-topic assignment,
  improve topic utilization and chemical breadth, and retain balanced ETM's
  broad MAG-optimizable beta inventory without importing another M1 mechanism.
- **Exact architectural/config change:** relative to the previously trained
  fragment/loss-balanced ETM, compute the normalized count-weighted document
  vector `u = normalize(x @ rho)`, then
  `g = softmax(2 * cosine(u, alpha) / 1.0)`, and use
  `normalize(theta_ETM * stopgrad(g) ** 1.0)` for every training and inference
  reconstruction. The ETM encoder, posterior, KL, fixed train-only 48D SGNS,
  50/50 fragment/loss beta, K=1000, seed, optimizer, batch size, and 120 epochs
  remain locked.
- **Why justified:** the synthetic three-seed screen improved held-out NLL,
  true beta recovery, topic use, and beta redundancy while preserving true
  theta recovery. The prior real balanced ETM already repaired optimized beta
  coverage (911 motifs), leaving diffuse theta and weak chemistry as the
  measured failure.
- **Training command:**
  `conda run --no-capture-output -n ms2lda-neural python -m scripts.run_msnlib_model_comparison train --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --method etm_balanced_gated --device cpu --etm-epochs 120 --etm-batch-size 256 --gate-temperature 1.0 --gate-gamma 1.0`
- **Chemical command:**
  `conda run --no-capture-output -n ms2lda-msnlib-mag python -m scripts.run_msnlib_model_comparison chemical --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --data-root /Users/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680 --method etm_balanced_gated_t1_g1`
- **Result:** training completed in 5,654.10 seconds with 19,278,000 parameters
  and finite outputs. Validation completion NLL was 8.695312 (OOV 0.031356).
  Chemistry found 890 optimized, 181 evaluable, and 108 useful motifs, with
  mean SOS 0.630221 and median SOS 0.629630; 1,020 spectra and 1,012 molecules
  met the locked `theta >= 0.5` association threshold. The median/mean
  effective topics per spectrum were 64.79/130.11, median maximum theta was
  0.3232, and 26.23% of spectra had maximum theta at least 0.5. Topic inventory
  broadened to 348 unique top-1 topics and a corpus effective topic count of
  489.55. Beta redundancy improved: mean nearest cosine was 0.3041, only one
  pair exceeded 0.99, no pair exceeded 0.999, and the largest strict duplicate
  component was one topic. Fragment probability mass remained 0.5 to numerical
  precision (median 0.4999999). Leakage filtering remained enabled.
- **Comparison with prior model:** relative to balanced ETM, NLL improved by
  0.070757, evaluable motifs increased by 15, useful motifs increased by 4,
  mean SOS increased by 0.000402, and beta/topic utilization improved. However,
  optimized motifs decreased by 21 and median effective topics worsened from
  46.72 to 64.79; high-confidence assignment also fell from 31.89% to 26.23%.
- **Decision:** the frozen candidate passes optimized, NLL, finite/stable, and
  duplicate-component gates, but fails evaluable (181 < 388), useful
  (108 < 252), and mean-SOS (0.630221 < 0.651498) gates. Run exactly one
  stronger-gate training experiment because the measured remaining failure is
  diffuse theta despite improved beta inventory. Do not test separation: the
  real model has no strict duplicate pair or component. Do not test NPMI: good
  assignment has not yet been established, so chemistry-only coherence is not
  the isolated failure.

## 2. One-dimensional stronger-gate diagnostic

- **Hypothesis:** increasing the detached gate exponent from 1 to 2 during
  training will reduce theta diffuseness and restore high-confidence document
  associations while retaining the improved topic inventory and completion.
- **Exact architectural/config change:** use `tau_g = 1.0`, `gamma = 2.0`; all
  other architecture, optimizer, seed, epoch, data, and evaluation settings are
  identical to experiment 1. This remains a separately trained model, not a
  post-hoc transformation.
- **Why justified:** experiment 1 improved beta utilization and redundancy but
  made theta more diffuse. At fixed geometry, temperature and exponent both
  control the same effective log-gate strength after renormalization, so only
  gamma is changed. The weaker direction is contradicted by the observed
  diffuseness, and no cross-product sweep is warranted.
- **Training command:**
  `conda run --no-capture-output -n ms2lda-neural python -m scripts.run_msnlib_model_comparison train --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --method etm_balanced_gated --device cpu --etm-epochs 120 --etm-batch-size 256 --gate-temperature 1.0 --gate-gamma 2.0`
- **Chemical command:**
  `conda run --no-capture-output -n ms2lda-msnlib-mag python -m scripts.run_msnlib_model_comparison chemical --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --data-root /Users/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680 --method etm_balanced_gated_t1_g2`
- **Result:** training completed all 120 epochs in 5,580.99 seconds with finite
  outputs and the same 19,278,000 parameters. Completion NLL improved to
  8.686967 (OOV 0.031356). Chemistry found 889 optimized, 220 evaluable, and
  138 useful motifs; mean/median SOS were 0.645148/0.642500, with SOS bands
  40 high, 98 intermediate, and 82 low. The locked threshold associated 877
  spectra and 869 molecules. Median/mean effective topics were 65.57/121.38,
  median maximum theta was 0.3024, and only 22.55% of spectra had maximum theta
  at least 0.5. Inventory broadened to 471 unique top-1 topics, 514 active
  topics above 0.0005, and a corpus effective count of 589.36. No pair exceeded
  beta cosine 0.999; maximum cosine was 0.99354 and the largest strict duplicate
  component was one. Fragment mass remained 0.5 to numerical precision.
- **Comparison with prior model:** relative to gamma 1, gamma 2 improves
  evaluable/useful motifs by 39/30, mean SOS by 0.014927, NLL by 0.008345,
  unique top-1 topics by 123, and corpus effective topic count by 99.81.
  However, it does not sharpen document assignment: median effective topics
  worsen from 64.79 to 65.57, median maximum theta falls from 0.3232 to 0.3024,
  and the fraction above theta 0.5 falls from 26.23% to 22.55%. Relative to
  balanced ETM it gains 54 evaluable and 34 useful motifs and 0.015329 mean
  SOS, but remains far below M1's 408/265 breadth and 0.658079 mean SOS.
- **Decision:** stop the gated-ETM study. Gamma 2 passes optimized, NLL,
  finite/stable, and no-catastrophic-duplicate gates, but fails evaluable
  (220 < 388), useful (138 < 252), and mean SOS
  (0.645148 < 0.651498). It is not near passing. A stronger gamma or post-hoc
  calibration would be another open-ended rescue rather than a bounded
  diagnostic. Separation remains unjustified because strict duplication is
  absent. NPMI remains unjustified because high-confidence document assignment
  is still poor. No candidate advances to test.

## Follow-up rule

The only follow-up was the pre-recorded one-dimensional gamma-2 experiment.
No separation, NPMI, post-hoc calibration, or additional gate-strength run was
performed. Separation remained conditional on a duplicate-component failure,
which neither gated model displayed; NPMI remained conditional on good
assignment with isolated weak chemistry, which was also not observed.

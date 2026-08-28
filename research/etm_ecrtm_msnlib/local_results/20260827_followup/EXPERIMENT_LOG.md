# Neural MS2LDA follow-up experiment log

Evidence boundary: validation only. Candidate test theta, completion, chemistry,
and result artifacts were not opened, loaded, scored, or summarized.

## 1. Artifact and M1 compatibility audit

- **Hypothesis:** the previous candidate artifacts and the locked M1 validation
  theta can be compared without rebuilding data.
- **Exact change:** none; read-only audit.
- **Why:** temperature calibration is only meaningful on identical validation
  records.
- **Config:** seed 42, K=1000, 3,889 validation spectra.
- **Command:** SHA-256 comparison of the two
  `data/validation_records.jsonl` files plus validation-theta shape checks.
- **Result:** both record files have SHA-256
  `0e85218489af6a07413474bb2db6ce74da537f6fd3c8ee77d5286f5775ba068c`;
  all compared theta arrays have shape `3889 x 1000`.
- **Interpretation:** locked M1 and follow-up mixture distributions are directly
  comparable.
- **Decision:** continue.

## 2. Pooled likelihood theta-temperature sweep

- **Hypothesis:** the pooled model's primary failure is inference calibration,
  so rank-preserving sharpening may recover broad locked-threshold chemistry.
- **Exact change:** no retraining and no beta change;
  `theta_new = normalize(theta_old ** (0.24 / tau_new))` in float64 log space.
- **Why:** the raw model had 967 optimized topics but only 14 evaluable topics.
- **Config:** tau grid
  `0.24, 0.20, 0.18, 0.16, 0.14, 0.12, 0.11, 0.10, 0.09, 0.08, 0.07, 0.06, 0.05, 0.04, 0.03`;
  locked association threshold 0.5; existing pooled MAG annotations reused.
- **Command:**
  `conda run -n ms2lda-neural python -m scripts.run_msnlib_neural_followup diagnose-existing --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --m1-run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/neural-minimality-seed42/m1-lock --output /Users/joewandy/Work/git/MS2LDA/research/etm_ecrtm_msnlib/local_results/20260827_followup --device cpu`
- **Result:** calibration steadily increases associations, but no tau passes the
  evaluable/useful gates. Tau 0.11 gives 293 evaluable, 194 useful, mean SOS
  0.665436, NLL 8.994812, and median 7.35 effective topics/spectrum versus M1's
  7.20. Tau 0.10 gives 305/205 and NLL 9.100824 but is sharper than M1. The
  useful count never exceeds 214 over the full grid.
- **Interpretation:** calibration is a real part of the failure, but not the
  whole failure. The robust chemistry/NLL region is approximately 0.10-0.12.
- **Decision:** freeze tau 0.11 as the M1-shape-matched diagnostic calibration;
  do not advance it because chemistry breadth still fails.

## 3. Pooled topic redundancy and competition

- **Hypothesis:** near-duplicate beta rows split or waste topic capacity and
  impose a hard ceiling on rank-preserving temperature calibration.
- **Exact change:** none; read-only beta, prototype, theta, and fixed-annotation
  diagnostics.
- **Why:** mean nearest beta cosine was 0.692 and maximum cosine was nearly 1.
- **Config:** cosine component thresholds 0.95, 0.99, and 0.999; top-1 and top-3
  validation usage; top-20 beta overlap.
- **Command:** included in the `diagnose-existing` command above.
- **Result:** only 374 topics ever win top-1, so 374 is the practical upper
  bound on evaluable topics under rank-preserving sharpening. A single
  614-topic component contains all 188,191 beta pairs at cosine >=0.999. Its
  prototype cosine median is 0.999996; all 614 topics are MAG-optimized, but
  none ever wins a spectrum. Their median beta has about 15,097 effective words,
  maximum word probability 0.000426, and only 0.00628 mass in its top 20 words.
  Weak MI=0.05 produces 615 near-duplicate topics and 372 distinct top-1 topics.
- **Interpretation:** the pooled model is both miscalibrated and severely
  component-collapsed. MAG optimization alone is not evidence of a useful
  topic when the beta row is almost uniform and the topic is never used.
- **Decision:** temperature-only rescue stops. A targeted published anti-collapse
  comparison or one explicit diversity mechanism is now scientifically
  justified; do not add a coherence loss because the identified defect is
  component duplication rather than top-word co-occurrence.

## 4. Canonical ETM temperature diagnostic

- **Hypothesis:** ETM's diffuse theta contributes to poor associations, while
  its fixed 609 optimized-topic ceiling prevents full rescue.
- **Exact change:** inference-only temperatures from 1.0 through 0.1; beta and
  MAG annotations unchanged.
- **Why:** separate theta calibration from beta/topic quality.
- **Config:** `1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1`.
- **Command:** included in `diagnose-existing`.
- **Result:** tau 0.8 increases evaluable/useful motifs from 130/79 to 163/100
  but raises NLL from 8.690730 to 9.242298. Stronger sharpening reaches at most
  186/114 and breaks the NLL gate from tau 0.6 downward.
- **Interpretation:** canonical ETM has a beta/topic-quality deficit that
  inference calibration cannot repair.
- **Decision:** stop canonical ETM calibration as a candidate; retain it only as
  a failure-mode diagnostic.

## 5. Fragment/loss-balanced ETM

- **Hypothesis:** fixed 50/50 decoder mass may repair real-data ETM's extreme
  fragment/loss topic skew and improve chemically annotatable beta coverage.
- **Exact change:** replace only the global vocabulary softmax with independent
  fragment and loss softmaxes, each assigned mass 0.5. Encoder, posterior, KL,
  fixed 48D SGNS, optimizer, seed, batch size, raw-count objective, and 120
  epochs remain paired with canonical ETM.
- **Why:** real ETM had 15.5% extreme channel-skew topics and only 609 optimized
  motifs.
- **Config:** seed 7043, K=1000, hidden=800, Adam 0.005, weight decay 1.2e-6,
  batch size 256, 120 epochs, CPU.
- **Command:**
  `conda run -n ms2lda-neural python -m scripts.run_msnlib_model_comparison train --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --method etm_balanced --device cpu --etm-epochs 120 --etm-batch-size 256`
- **Smoke result:** finite objective and gradients at real K/V; fragment and loss
  masses equal 0.5 within floating-point tolerance.
- **Training result:** complete and finite. Completion NLL is 8.766069; median
  effective topics/spectrum is 46.72; corpus effective topics is 355.71;
  active topics above 0.0005 / at least 1/K are 275 / 167; mean nearest beta
  cosine is 0.39961; maximum pairwise beta cosine is 0.99583; top-word
  uniqueness is 0.3356. Fragment mass is 0.5 within numerical precision for
  every topic and extreme channel skew falls from 15.5% to 0%.
- **Chemistry result:** 911 optimized, 166 evaluable, 104 useful, mean SOS
  0.629819, median SOS 0.632929, and 1,240 / 1,232 associated validation
  spectra / molecules. SOS bands are 19 high, 85 intermediate, and 62 low.
- **Interpretation:** channel balancing repairs the optimized-motif deficit
  (609 -> 911) and removes channel skew, but raw theta remains diffuse and
  chemistry quality/breadth still fails. Topic redundancy is not clearly
  improved.
- **Decision:** because beta coverage now passes and diffuse theta is the
  remaining association bottleneck, run exactly one post-hoc balanced-ETM
  temperature sweep. Only 260 topics ever win top-1 on validation, so
  rank-preserving calibration cannot reach the 388-evaluable gate; the sweep
  is diagnostic and cannot by itself rescue the model. Do not retrain or add
  another mechanism.
- **Temperature result:** tau 0.8 is the only sharpened point that retains the
  NLL gate (NLL 9.294272), but gives only 209 evaluable / 130 useful motifs and
  mean SOS 0.620555. Tau 0.6 and below break the NLL gate. The hardest settings
  asymptote at 240 evaluable / 148 useful while producing nearly one-topic-per-
  spectrum mixtures and NLL 13-14; mean SOS never passes at any point.
- **Temperature command:**
  `conda run --no-capture-output -n ms2lda-neural python -m scripts.run_msnlib_neural_followup sweep-etm-temperature --run /Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42 --method etm_balanced --output /Users/joewandy/Work/git/MS2LDA/research/etm_ecrtm_msnlib/local_results/20260827_followup --device cpu`.
- **Temperature decision:** stop. No balanced-ETM temperature is selected or
  presented as a candidate. Sharpening cannot repair its topic-usage ceiling or
  reduced chemical quality.

## 6. Canonical ECRTM anti-collapse follow-up

- **Hypothesis:** a published embedding-clustering topic model can preserve a
  non-duplicate topic inventory where the pooled prototype bank collapsed.
- **Exact change:** use the maintained TopMost-style ECRTM equations and the
  canonical Sinkhorn stopping rule; do not transplant ECR into ETM or add an
  MS-specific mechanism.
- **Why:** 614/1000 pooled prototypes form one near-exact component and never
  win a validation spectrum. This is direct component-collapse evidence and
  satisfies the handoff's conditional ECRTM trigger.
- **Config:** K=1000, V=21,233, train-only 48D SGNS initialization, ECR weight
  100, Sinkhorn alpha 20, residual tolerance 0.005, maximum 1,000 iterations,
  Adam 0.002, batch size 200, 40 epochs, seed 8043. Evaluate raw theta and the
  previously documented inference-only tau 0.30 calibration.
- **Feasibility evidence:** the real K/V canonical probe converged in 201
  iterations with residual 0.004747 and 4.659 seconds per batch, projecting
  about 7.1 hours for 40 epochs. The 50-step approximation was not converged.
- **Preflight:** CLI routing, 19 initial focused tests, formatting/lint, and a synthetic
  uninterrupted-versus-checkpoint-resumed comparison passed. The resumed run
  was bit-identical (`maximum absolute parameter difference = 0.0`). Before the
  long run, the solver was hardened to reject non-finite residuals explicitly,
  and the checkpoint contract was bound to SHA-256 hashes of the training
  matrix, token features, and protocol plus optimizer/ECR settings. These are
  operational safeguards and do not change the published equations. The
  hardened path then passed 20 focused tests.
- **First full-run epoch:** complete in 245.499 seconds; mean / maximum
  Sinkhorn iterations 54.65 / 201; mean / maximum checked residual 0.000488 /
  0.004974. By epochs 5-6, the solver stabilized at 151 iterations/batch and
  about 688-716 seconds/epoch, with maximum residual below 0.00466. The
  canonical tolerance remains met; the updated projection is roughly 7-9
  hours for 40 epochs.
- **Full-run result:** operationally infeasible under the canonical solver.
  Twenty-one of 40 epochs completed, representing 21,971.14 seconds (6.10 h)
  of completed-epoch compute. Sinkhorn demand rose from 54.65 mean iterations
  in epoch 1 to 721.07 mean / 951 maximum in epoch 21; epoch 21 alone took
  3,321.50 seconds. During epoch 22, at least one batch failed to reach the
  locked residual tolerance 0.005 within 1,000 iterations. A single exact
  checkpoint resume restored epoch 21 state/RNG and failed again at the same
  boundary.
- **Failure handling:** fail closed. No partial ECRTM beta/theta was inferred,
  no partial model was sent to MAG/SOS, and neither raw nor tau 0.30 chemistry
  was produced. Checkpoint SHA-256 is
  `c1220c0785c4dd36fc264b20f7b66cdca3c784b04fd8f6e9c3c1120060596976`.
- **Decision:** canonical ECRTM was scientifically warranted, but this full
  K/V optimization is numerically and operationally unsuitable here. Stop;
  do not raise the iteration cap into an unbounded runtime escalation and do
  not substitute the known-unconverged 50-step approximation.

## Operational notes

- The first diagnostic invocation stopped before a sweep because the report
  name `nll_per_in_vocab_token` was used instead of the scorer's canonical
  `nll_per_token` key. A regression test was added.
- The second invocation stopped after writing the pooled sweep because a
  read-only memory-mapped beta view was normalized in place. The diagnostic now
  makes an explicit working copy. No model or locked artifact was changed by
  either failure.
- The follow-up reuses the already-built shared leakage-filtered MAG index.
  That index excludes compound identifiers from both held-out splits, but this
  campaign does not load or score candidate test theta, test completion, test
  chemistry, or test result artifacts. Candidate selection is validation-only.

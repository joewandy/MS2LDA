# Instructions for the higher-compute implementing agent

## Mission

Run a **validation-only real MSnLib comparison** of recognizable published neural topic models before adding any new MS-specific architecture.

Primary candidates:

1. fixed-pretrained-SGNS ETM;
2. maintained TopMost-style ECRTM initialized from the same SGNS vectors;
3. ECRTM with the already-predeclared inference-only theta temperature `tau=0.30`, evaluated as a calibration of the same trained ECRTM rather than a separately trained model.

Control/reference:

- committed M1 result at branch base `20de0e45aec25203e6bc38770a795b25cc18bff7`.

Do **not** redesign the model while running this campaign.

## Non-negotiable evidence boundary

- Candidate selection uses **validation only**.
- Do not inspect candidate test chemistry, candidate test theta, or candidate test completion metrics until a candidate has passed the predeclared validation gates.
- M1's already committed test result can be cited as historical context, but it must not guide candidate tuning.
- Do not tune theta temperature on MSnLib test. `tau=0.30` is already frozen from synthetic seeds 11/23/37.
- Do not add fragment/loss balancing, NPMI, prototype separation, document gating, sparse routing, DreaMS, or any other new mechanism during the first run.

## Data and preprocessing

Use the repository's existing frozen public acquisition and preprocessing code, not a new parser.

Acquire the full assets with:

`python scripts/download_msnlib_validation_assets.py --data-root <DATA_ROOT>`

This verifies the frozen Zenodo record and provides both:

- `Data/Benchmark_MSn_Lib/Corinna_Library_filtered_positive.mgf`;
- the positive-mode Spec2Vec database/model/embedding assets required for leakage-filtered MAG.

Then use the existing repository preparation so that all methods share exactly:

- seed 42;
- scaffold/compound-disjoint 70/10/20 split;
- train-only vocabulary;
- K=1000;
- V determined by the locked train vocabulary (historically 21,233);
- intensity pseudo-count representation;
- train-only 48-dimensional SGNS embeddings;
- exact peak-pair-safe document completion split;
- exact validation records and chemistry evaluation.

The research runner `scripts/run_published_topic_models_msnlib.py` already calls the repository's `prepare_data()` and `train_token_features()` rather than rebuilding these pieces independently.

## Model A: fixed-SGNS ETM

Use the original ETM architecture/ELBO as represented in the research runner:

- fixed train-only 48-dimensional SGNS word embeddings;
- two-layer ReLU document encoder;
- Gaussian variational posterior;
- logistic-normal theta;
- standard-normal KL;
- embedded topic-word decoder;
- raw count reconstruction, normalized BOW encoder input;
- no MS-specific fragment/loss balancing;
- no ECR/Sinkhorn/NPMI/prototype-separation/gating.

Starting training settings are deliberately inherited from the original ETM code rather than tuned on MSnLib:

- Adam;
- learning rate 0.005;
- weight decay 1.2e-6;
- hidden size 800;
- 120 epochs;
- batch size 256 in the research runner.

If memory requires a smaller batch size, that may be changed as an operational parameter, but record it and do not use validation chemistry to choose it.

## Model B: ECRTM

Use the maintained TopMost equations, initialized from the same train-only SGNS embeddings.

Reference characteristics:

- 200-unit two-layer softplus encoder;
- logistic-normal approximation to symmetric Dirichlet prior;
- trainable word embeddings initialized from SGNS;
- trainable topic embeddings;
- beta temperature 0.2;
- ECR weight 100 (maintained TopMost default used in the synthetic screen);
- Sinkhorn alpha 20;
- Adam learning rate 0.002;
- batch size 200 unless memory requires otherwise.

### First do a scalability probe

Do not blindly launch 40 full epochs at K=1000, V~21k.

ECR builds a full K x V transport geometry and the published minibatch trainer repeatedly evaluates the ECR objective. At real scale this may be expensive even on a GPU.

Before full training:

1. instantiate the full K=1000, V=real model;
2. run a small number of real training batches, including ECR forward/backward;
3. record peak memory, wall time per batch and projected epoch time;
4. run the published/exact stopping rule where feasible on this probe;
5. compare with the already-studied bounded 50-step numerical approximation.

If the canonical solver is operationally practical, prefer it.

If it is not practical, the 50-step approximation is an acceptable research candidate because the synthetic exact-vs-50 comparison was nearly identical. Label it explicitly as a bounded numerical approximation and retain the exact-solver feasibility measurements.

Do not change the ECR mathematical objective to rescue runtime in this first campaign.

## ECRTM theta calibration

Evaluate the trained ECRTM twice on validation:

1. raw deterministic theta (`tau=1`);
2. frozen synthetic-derived calibration:

`theta_tau = softmax(log(theta) / 0.30)`.

Do not retrain for the calibrated result. Beta and model weights are identical.

The synthetic reason for evaluating this is that raw ECRTM kept topics distinct but spread a 1-3-motif spectrum over roughly 30 effective topics at K=36. `tau=0.30` reduced that to about 4.7 effective topics and improved truth-known theta recovery across held-out synthetic seeds.

On real validation, report both raw and calibrated results. Do not choose a new tau.

## Completion and collapse diagnostics

For every candidate, record at least:

- validation completion NLL per in-vocabulary token;
- OOV completion fraction;
- median effective topic count per full spectrum `exp(H(theta_d))`;
- corpus effective topic count;
- number of materially active topics using both:
  - mean usage >= 1/K;
  - mean usage > 0.0005;
- maximum mean topic usage;
- mean nearest-topic beta cosine;
- maximum pairwise beta cosine;
- fit wall time;
- inference throughput if feasible;
- peak memory.

These diagnostics matter because the synthetic ETM collapse had good NLL while using only a small fraction of topics.

## Chemical validation: use the exact existing MAG/SOS machinery

The decisive result is not NLL alone.

Use the candidate `beta.npy` and validation full-spectrum theta with the repository's existing leakage-filtered MAG/SOS implementation in `benchmarks/neural_ms2lda/chemical.py` and `mag.py`.

The research runner already writes candidate artifacts under:

- `validation_evaluation/etm/`;
- `validation_evaluation/ecrtm/`;
- `validation_evaluation/ecrtm_tau030/` when ECRTM finishes.

The core `run_chemical_scoring()` currently whitelists only `neural` and `tomotopy`. For this isolated research branch, minimally generalize that whitelist or add a thin research wrapper so `etm`, `ecrtm` and `ecrtm_tau030` pass through the **same** `_shared_annotations()` and `_topic_scores()` machinery. Do not reimplement MAG or SOS.

For ECRTM topic spectra, use the published/TopMost topic-word weights from the ECR geometry (`get_beta` / equivalent). Row normalization is allowed because it does not change within-topic word rankings or cosine geometry. Keep the decoder-completion calculation separate and exact to ECRTM's `decoder_bn(theta @ beta)` formulation.

Report:

- annotation coverage / optimized motifs;
- high-confidence evaluable motifs;
- useful motifs (SOS >= 0.6 under the existing bands);
- mean SOS;
- median SOS;
- SOS band counts;
- associated spectra/molecules;
- leakage audit status.

## Provisional validation gates

Compare against the committed M1 validation result:

- optimized motifs: 884;
- evaluable motifs: 408;
- useful motifs: 265;
- mean SOS: 0.6580793714;
- median SOS: 0.6488636364;
- completion NLL: 8.9741399256.

Candidate gate:

- optimized motifs >= 840 (95% of M1);
- evaluable motifs >= 388 (95%);
- useful motifs >= 252 (95%);
- mean SOS >= 0.651498 (99%);
- completion NLL <= 9.422847 (105%);
- finite/stable execution required.

If an existing repository selection rule is stricter and directly applicable, preserve the stricter locked rule.

### Interpretation of outcomes

**If fixed-SGNS ETM passes:**

Prefer it scientifically unless ECRTM yields a clear chemistry advantage. It is the cleanest paper story: classical MS2LDA -> published Embedded Topic Model using spectral SGNS embeddings.

**If ETM retains useful motifs but has obvious duplicate/starved topics:**

ECRTM is directly motivated as a published anti-collapse replacement.

**If ECRTM raw theta gives almost no spectra above the existing 0.5 membership threshold but tau=0.30 recovers chemistry:**

Report raw and calibrated results. The next methodological question becomes theta calibration / count scaling, not a new topic architecture.

**If both ETM and ECRTM fail SOS while NLL is good:**

Do not immediately restore all M1 machinery. The most plausible missing ingredient is chemistry/co-occurrence structure. Test one established or clearly motivated addition at a time, starting with positive-NPMI/co-occurrence coherence only if the failure pattern supports it.

**If ECRTM is computationally infeasible at K=1000:**

Treat that as a valid operational result. Do not distort the model solely to force completion. Fixed-SGNS ETM can remain the main published-model candidate.

## Test split rule

Do not open candidate test data until one candidate passes validation.

After a validation pass:

1. freeze model form and all hyperparameters;
2. preferably repeat the candidate over at least five seeds on training+validation selection protocol if compute allows, quantifying beta/motif stability;
3. then perform exactly one test evaluation for reporting;
4. compare with the already committed M1 and Tomotopy test results without reopening architecture selection.

## Files to read before running

1. `research/etm_ecrtm_msnlib/README.md` — scientific findings and rationale.
2. `scripts/run_published_topic_models_msnlib.py` — prepared real-data ETM/ECRTM runner.
3. `benchmarks/neural_ms2lda/protocol.json` — locked MSnLib constants.
4. `benchmarks/neural_ms2lda/data.py` / `spectra.py` — exact split, vocabulary and SGNS construction.
5. `benchmarks/neural_ms2lda/chemical.py` / `mag.py` — exact chemistry scoring.
6. `benchmarks/neural_ms2lda/results/seed42/results.json` — incumbent M1/Tomotopy evidence.
7. `research/etm_ecrtm_msnlib/repro/` — synthetic truth-known reproduction code.
8. `research/etm_ecrtm_msnlib/results/` — exact synthetic CSV evidence.

## What not to do

- Do not merge this branch to main merely because the runner executes.
- Do not call the synthetic results real chemical validation.
- Do not tune K; keep K=1000 for the direct comparison.
- Do not tune tau on the test split.
- Do not add DreaMS in this campaign.
- Do not add fragment/loss channel balancing in the first ETM/ECRTM run.
- Do not invent an ETM+ECR hybrid; it already failed the synthetic negative-control experiment.
- Do not judge success from completion NLL alone.
- Do not delete or overwrite the locked M1 evidence.

The desired output from the compute agent is a compact validation report plus machine-readable result files sufficient to decide whether ETM, ECRTM, or neither should become the paper-facing neural model.

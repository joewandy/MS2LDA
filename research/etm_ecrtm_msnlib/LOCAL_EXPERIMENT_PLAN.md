# Consolidated local MSnLib experiment plan

## Goal

Run a **validation-only real MSnLib comparison** that answers how far Neural MS2LDA can be simplified while preserving chemically useful Mass2Motifs.

Do not open candidate test results in this campaign. Push the validation evidence to a separate GitHub experiment branch for review before any test evaluation.

## Repository starting point

The canonical research handoff branch is:

`research/etm-ecrtm-msnlib-20260826`

At the time this plan was frozen, it contains current `main` as an ancestor and is 0 commits behind `main`. It contains:

- `HANDOFF.md`, `NEXT_AGENT.md`, `LITERATURE_SURVEY.md`, `REFERENCES.bib`;
- `CHANNEL_BALANCE_SIMULATION.md`;
- `scripts/run_published_topic_models_msnlib.py`;
- ETM/ECRTM synthetic evidence under `results/` and `repro/`;
- the separate pooled-projected reference model under `pooled_projected/`.

Before doing any scientific work, fetch both `main` and the research branch and make a **new local execution branch** from the research branch. Merge current `origin/main` into that execution branch if `main` has advanced. Do not work directly on `main` or directly on the handoff branch.

Suggested execution branch:

`experiment/msnlib-etm-pooled-local-20260827`

If that name already exists, append a short machine/date suffix.

## Read first

In this order:

1. `research/etm_ecrtm_msnlib/LOCAL_EXPERIMENT_PLAN.md`
2. `research/etm_ecrtm_msnlib/HANDOFF.md`
3. `research/etm_ecrtm_msnlib/CHANNEL_BALANCE_SIMULATION.md`
4. `research/etm_ecrtm_msnlib/pooled_projected/README.md`
5. `research/etm_ecrtm_msnlib/NEXT_AGENT.md`
6. `research/etm_ecrtm_msnlib/LITERATURE_SURVEY.md`
7. `scripts/run_published_topic_models_msnlib.py`
8. the locked repository protocol/data/chemical-scoring code under `benchmarks/neural_ms2lda/`.

## Evidence boundary

- Use the exact existing MSnLib split, train-only vocabulary, train-only SGNS, completion views, MAG and SOS machinery.
- Candidate selection is **validation only**.
- Do not inspect candidate test completion, test theta, test chemistry or test motif scores.
- Do not tune on test.
- Do not overwrite M1 artifacts or its locked result ledger.
- Do not redesign a model while evaluating it.

## Primary first-wave models

### A. Fixed-SGNS canonical ETM

Use the faithful ETM implementation already in `scripts/run_published_topic_models_msnlib.py`:

- fixed train-only 48D SGNS embeddings;
- original two-layer ReLU variational encoder;
- Gaussian posterior / logistic-normal theta;
- standard-normal KL;
- canonical embedded ETM decoder with **global topic-word softmax**;
- raw count reconstruction, normalized BOW encoder input;
- no fragment/loss balancing;
- no ECR, NPMI, prototype separation, routing or document gate.

Starting settings remain the original ETM-derived values already documented in the handoff. Operational batch-size changes are allowed for memory, but record them.

### B. Pooled Projected MS2LDA, likelihood only

Use/adapt:

- `pooled_projected/simple_candidate.py`
- `pooled_projected/simple_candidate_training.py`
- `pooled_projected/protocol_minimum.json`

This is the strongest minimal custom candidate from the separate simplification study. Preserve its studied 50/50 fragment/loss decoder for this comparison; do not reinterpret the later ETM channel-balance result as a reason to modify the pooled candidate before real validation.

### C. Pooled Projected MS2LDA + weak MI

Use the identical model with `pooled_projected/protocol_mi005.json` (`mi_weight=0.05`). Treat this as a secondary diagnostic candidate. The likelihood-only model remains scientifically preferable if both perform equivalently.

### Reference: M1

Use the existing committed M1 validation evidence and exact same scoring contract as the reference. Retraining M1 is not required unless needed to verify the local environment; do not change M1.

## Conditional models, not first-wave requirements

### ECRTM

Run a full-K/V feasibility probe after the primary candidates. Proceed to full ECRTM only if:

- fixed-SGNS ETM exhibits material topic starvation/duplication/collapse; or
- compute is clearly practical and the extra published anti-collapse comparator will not delay the primary validation result.

Prefer the canonical solver when practical; otherwise the documented 50-step Sinkhorn cap may be used only as an explicitly labelled numerical approximation.

### ETM + forced 50/50 channel balance

Do **not** run by default. The paired simulation screen found no consistent benefit for fixed-SGNS ETM. First record the real ETM fragment-mass distribution. Consider a balanced ETM follow-up only if real ETM shows a material channel-skew pathology.

## Data acquisition

Use the repository script:

`python scripts/download_msnlib_validation_assets.py --data-root <DATA_ROOT>`

Use the full assets needed for leakage-filtered MAG/SOS, not only the MGF. Verify downloads with the repository's existing checks.

Then use the repository's preparation code rather than reimplementing parsing, splitting, vocabulary or SGNS.

Expected locked characteristics include seed-42 scaffold/compound-disjoint 70/10/20 split, K=1000, train-only vocabulary (historically V≈21,233), train-only 48D SGNS and peak-pair-safe completion split.

## Chemical scoring

This is the decisive evaluation. Route every candidate through the same existing `benchmarks/neural_ms2lda/chemical.py` / `mag.py` logic used by M1. Minimally extend method-name plumbing if needed; do not reimplement MAG/SOS.

For each candidate save:

- optimized/annotated motifs;
- high-confidence evaluable motifs;
- useful motifs under the existing SOS threshold;
- mean and median SOS;
- SOS band counts;
- associated spectra and molecules;
- leakage audit status.

## Completion and collapse diagnostics

Save at least:

- validation completion NLL per in-vocabulary token;
- OOV completion fraction;
- median effective topics per spectrum `exp(H(theta_d))`;
- corpus effective topic count;
- active-topic counts using several documented thresholds, including mean usage >0.0005 and >=1/K;
- maximum mean topic usage;
- mean nearest-topic beta cosine;
- maximum pairwise beta cosine;
- top-word uniqueness;
- fragment probability mass per topic (especially for ETM);
- fit wall time;
- inference throughput;
- peak CPU/GPU memory where practical;
- finite/stability status.

For pooled + MI also save conditional theta entropy, marginal theta entropy and their difference separately.

## Frozen provisional validation gates

Reference M1 validation values:

- optimized motifs: 884
- evaluable motifs: 408
- useful motifs: 265
- mean SOS: 0.6580793714
- median SOS: 0.6488636364
- completion NLL: 8.9741399256

Candidate first-pass gate:

- optimized motifs >= 840;
- evaluable motifs >= 388;
- useful motifs >= 252;
- mean SOS >= 0.651498;
- completion NLL <= 9.422847;
- finite/stable execution;
- no obvious catastrophic inventory collapse.

If an existing locked repository rule is stricter and directly applicable, use the stricter rule.

Do not rescue a chemistry failure merely because NLL is better.

## Result directory and what to commit

Create a result directory such as:

`research/etm_ecrtm_msnlib/local_results/<run_id>/`

Commit **text/small machine-readable evidence**, including:

- `README.md` with concise findings and exact commands;
- `environment.txt` or equivalent package/GPU/OS snapshot;
- `provenance.json` with base commit, execution commit, data/checksum identifiers and exact configs;
- per-model `metrics.json`;
- per-model `training_history.csv`;
- per-model `chemical_scores.csv` / motif-level MAG/SOS table where size permits;
- `comparison.csv` with all headline metrics;
- top-word/topic summaries (for example top 20 words per topic);
- ETM fragment-mass distribution summary;
- runtime/memory measurements;
- failure logs if any candidate is operationally infeasible.

Do **not** commit huge raw `beta.npy`, model checkpoints, downloaded MSnLib assets or multi-GB databases. Keep those locally and record their local filename, byte size and SHA-256 in provenance. If a small compressed theta artifact is useful it may be retained locally; the text summaries above are the required GitHub review surface.

## Push-back workflow

When validation work is complete:

1. run tests/sanity checks;
2. ensure no large data/model assets are staged;
3. commit all implementation changes, configs and result summaries to the execution branch;
4. `git push -u origin <execution-branch>`;
5. report the exact branch name and final commit SHA to the user.

Do not merge to `main` and do not open candidate test data. The user will bring the pushed branch back to ChatGPT for independent analysis before deciding the next experiment or test evaluation.

## Desired decision output

The validation report should answer:

1. Does canonical fixed-SGNS ETM preserve enough chemically useful motif breadth to replace M1?
2. Does the pooled projected likelihood-only model preserve chemistry while substantially simplifying inference/training?
3. Does weak MI materially help the pooled model, or is it unnecessary?
4. Does ETM show actual topic collapse or fragment/loss channel pathology on real MSnLib?
5. Is ECRTM warranted at all?
6. Which model, if any, should advance to a frozen multi-seed stability check and then a single locked test evaluation?

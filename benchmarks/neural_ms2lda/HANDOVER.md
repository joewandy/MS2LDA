# Neural MS2LDA handover

This document is the starting point for the next research agent. It describes
the single supported model, where its evidence lives, how to reproduce it, and
how to conduct the next ablation study without contaminating the held-out test
comparison. Read the scientific report before changing the model.

## Current state

The repository contains one neural MS2LDA architecture. It is a seed-42,
K=1000 research result trained for 40 epochs with six CPU threads. Inference is
one forward routing pass; there is no per-spectrum optimization. Tomotopy is an
independently trained comparator, not a teacher.

The cleanup deliberately removed alternative architectures, compatibility
branches, artifact schemas, checkpoint selection, provenance manifests, report
hashes, and rejected experiments. Do not restore them. Start work from the
current `main`, confirm that it matches `origin/main`, and create a fresh
`codex/` topic branch.

This is a single-dataset, single-seed research result. It is not evidence that
the production Tomotopy backend should be replaced.

## Authoritative files

| Purpose | Location |
| --- | --- |
| Complete mathematical and scientific description | `docs/research/neural_ms2lda_report.tex` |
| Compiled report | `docs/research/neural_ms2lda_report.pdf` |
| Fixed study constants | `benchmarks/neural_ms2lda/protocol.json` |
| Forward model and one-pass inference | `benchmarks/neural_ms2lda/model.py` |
| Likelihood and regularizers | `benchmarks/neural_ms2lda/objectives.py` |
| Optimization and recycling | `benchmarks/neural_ms2lda/training.py`, `optimization.py` |
| Data preparation and deterministic views | `benchmarks/neural_ms2lda/spectra.py`, `data.py` |
| Held-out evaluation | `benchmarks/neural_ms2lda/evaluation.py` |
| MAG annotation and SOS | `benchmarks/neural_ms2lda/mag.py`, `chemical.py` |
| End-to-end orchestration | `benchmarks/neural_ms2lda/pipeline.py`, `__main__.py` |
| Trained model | `benchmarks/neural_ms2lda/results/seed42/trained_model/` |
| Canonical numerical evidence | `benchmarks/neural_ms2lda/results/seed42/results.json` |
| Generated report fragments | `docs/research/generated/` |
| Report generator | `scripts/generate_neural_ms2lda_report.py` |

`results.json` is the only numerical source for the paper. The trained model
contains only `weights.pt`, `model.json`, and `vocabulary.json`.

On the current workstation, the acquired public inputs are under:

```text
/Users/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680
```

That path is not part of the scientific model. A new machine may use any data
root with the same public Zenodo inputs. Each run directory is bound to the
resolved data root used when it is created, preventing mixed-input resumes.

## Implemented model

The report gives the exact equations. The essential implementation is:

1. Build a fixed 64-dimensional token feature from train-only SGNS,
   sinusoidal mass coordinates, and fragment/loss indicators.
2. Project tokens to normalized 128-dimensional vectors and learn 1000
   normalized topic prototypes in the same geometry.
3. Decode topic words with cosine logits. Fragment and neutral-loss channel
   evidence uses log-mean-exp, followed by a fixed 0.25 pull toward equal
   channel mass.
4. Route each observed token using its projected vector, a leave-one-out
   document context MLP, and additive whole-spectrum topic evidence.
5. Retain and renormalize the top two topics per token. Training uses the
   straight-through estimator; inference uses the same hard forward values.
6. Aggregate count-weighted token mass and multiply it by detached
   whole-spectrum evidence raised to 0.75. Exact zero support is preserved; an
   empty spectrum receives a uniform mixture.
7. Decode a spectrum with `theta @ beta` and optimize count-weighted
   document-completion likelihood.

Training alternates router and topic blocks. Its anti-collapse mechanisms are
paired partial views, Jensen-Shannon view agreement, Sinkhorn usage targets,
local token reconstruction, a positive-NPMI graph, prototype separation,
weighted spherical k-means++ initialization, temperature annealing, and
deterministic dead-topic recycling. These mechanisms have different intended
roles, but they have not yet been isolated by a complete ablation study.

## Accepted evidence

The paper-aligned validation baseline is:

| Metric | Neural | Tomotopy |
| --- | ---: | ---: |
| Optimized motifs / MAG coverage | 663 / 66.3% | 607 / 60.7% |
| High-confidence SOS-evaluable motifs | 312 | 206 |
| Useful high-confidence motifs, SOS >= 0.6 | 185 | 138 |
| Mean high-confidence SOS | 0.6323 | 0.6761 |
| Median high-confidence SOS | 0.6364 | 0.6854 |
| Completion NLL per token | 8.5014 | 9.6622 |
| Fitting time | 4719.98 s | 4561.68 s |

The test comparison is already public and therefore cannot be made unknown
again. Nevertheless, it must not be read, ranked, or used to choose ablations.
The next study must remain validation-only until one final architecture is
locked. The committed test result is confirmation evidence, not a development
target.

The model weights, full topic-word matrix, fixed-batch document mixtures, paper
metrics, and completion NLL were checked for exact equality through the cleanup.
The later evaluation fix changes only which already-computed mixture supplies a
secondary full-spectrum entropy diagnostic; it does not alter model parameters
or paper metrics.

## Reproduction and verification

Create the two study environments, acquire the public data, and run:

```bash
conda env create -f environment-neural-ms2lda.yml
conda env create -f environment-msnlib-mag.yml

conda run -n ms2lda-neural python \
  scripts/download_msnlib_validation_assets.py \
  --data-root /path/to/MSnLib-assets

conda run -n ms2lda-neural python -m benchmarks.neural_ms2lda run \
  --data-root /path/to/MSnLib-assets \
  --run /path/to/new-run
```

Use `status --run /path/to/new-run` for progress. An interrupted neural fit
restarts from deterministic initialization; completed stages may be reused only
with the original bound data root.

Before handing off a change, run:

```bash
black --check benchmarks/neural_ms2lda \
  scripts/download_msnlib_validation_assets.py \
  scripts/generate_neural_ms2lda_report.py

ruff check --select E,F,I benchmarks/neural_ms2lda \
  scripts/download_msnlib_validation_assets.py \
  scripts/generate_neural_ms2lda_report.py

pytest -q benchmarks/neural_ms2lda/tests
python scripts/generate_neural_ms2lda_report.py
test -z "$(git status --porcelain --untracked-files=all -- docs/research/generated)"
```

Also run the upstream regression suite with `NUMBA_DISABLE_JIT=1`, build and
inspect the production wheel, compile the LaTeX report, and visually inspect
every PDF page. Benchmark research code must remain excluded from the wheel.

## Next assignment: thorough simplification ablation

Do not start this study while merely reviewing this handover. The next active
research task is to determine which model components are actually necessary
and reduce the final implementation to the smallest architecture that retains
the accepted performance.

### Experimental discipline

- Create `codex/neural-ms2lda-ablation-seed42` from the then-current `main`.
- Use seed 42, K=1000, six CPU threads, epoch 40, the current public data,
  deterministic split, vocabulary, views, and initialization.
- Reuse the locked Tomotopy values. Do not retrain Tomotopy for neural
  ablations.
- Build a temporary validation-only driver that cannot open test matrices or
  test records. Add a regression test proving that absence.
- Retrain every candidate from deterministic initialization. Do not reinterpret
  the committed weights under a changed forward model.
- Change one component at a time in the first phase. Do not perform a broad
  hyperparameter sweep to rescue a failed removal.
- Record runtime but do not accept a scientifically worse model merely because
  it is faster.
- Keep experimental switches and rejected paths on the ablation branch only.
  Once a final architecture is chosen, make it unconditional and delete the
  ablation machinery before proposing integration.

### Validation retention gate

Predeclare the following gate for a simpler candidate:

- MAG annotation coverage at least 65.0%, or at least 650 optimized motifs.
- At least 296 high-confidence SOS-evaluable motifs, 95% of the current 312.
- At least 176 useful high-confidence motifs, 95% of the current 185.
- Mean high-confidence SOS at least 0.6223, within 0.01 of the current mean.
- Validation completion NLL at most 8.5266.
- Finite training and evaluation with no clear topic-collapse failure.

A component is provisionally dispensable only when its removal passes every
gate. It is necessary when removal fails a gate by a material margin. Mark
borderline outcomes as ambiguous and revisit them in the interaction phase;
do not decide from one favourable metric.

### Phase 0: establish the control

Run the complete current model once in the ablation environment. Confirm that
its validation metrics reproduce the committed baseline closely enough to use
the retention gate. If the control misses the gate, diagnose the environment
or data before testing components.

### Phase 1: single-component removals

Evaluate each removal independently. Order experiments by likely code and
conceptual simplification:

| ID | Removal or simplification | Main code area | Question |
| --- | --- | --- | --- |
| A1 | Remove the 0.75 document gate | `model.aggregate_theta` | Is token mass alone sufficient? |
| A2 | Remove additive document evidence from token logits | `model.route` | Is local contextual routing sufficient? |
| A3 | Remove the context MLP, using projected tokens directly | `model._route_embeddings` | Does learned context justify its parameters? |
| A4 | Route each token to top one instead of top two | `model.route` | Is the second assignment needed? |
| B1 | Remove the 0.25 equal-channel pull | `model.topic_word_distribution` | Does mean evidence already balance token types? |
| B2 | Replace channel-aware decoding with one global softmax | `model.topic_word_distribution` | Is the entire channel correction necessary? |
| C1 | Remove Jensen-Shannon view consistency | `objectives.router_block_loss` | Does paired completion already align views? |
| C2 | Remove local decoder reconstruction | `objectives.topic_block_loss` | Is cross-view completion sufficient? |
| C3 | Remove the positive-NPMI loss | `objectives.topic_block_loss` | Is train-only graph structure still needed? |
| C4 | Remove prototype separation | `objectives`, `optimization` | Do routing and recycling maintain distinct topics? |
| C5 | Remove Sinkhorn targets | `objectives.router_block_loss` | Is explicit early usage pressure necessary? |
| C6 | Remove dead-topic recycling | `optimization.validate_and_recycle` | Can initialization and losses maintain capacity? |
| C7 | Replace weighted spherical k-means++ with the simplest deterministic seeding | `model.initialize_model` | Is elaborate initialization necessary? |
| D1 | Remove SGNS coordinates from token features | `data.build_token_features` | Are mass and token type features sufficient? |
| D2 | Remove Fourier mass coordinates | `data.build_token_features` | Does SGNS plus type encode enough mass structure? |
| E1 | Remove paired partial views and train on full spectra | `data`, `training`, `objectives` | Is the corruption/completion construction essential? |
| E2 | Replace alternating router/topic blocks with one joint step | `training`, `optimization` | Does alternating optimization justify its complexity? |

Do not combine B1 and B2: B2 subsumes B1. Treat D1, D2, E1, and E2 as
late single removals because they alter broad parts of the training problem.

### Phase 2: backward elimination

Rank all passing single removals by simplification value: eliminated learned
parameters, deleted training stages, deleted data products, and code that can
be removed. Starting from the highest-value removal, build a cumulative model
one removal at a time. Retrain and reapply the full validation gate after every
step. Stop adding a removal as soon as it breaks the gate; retain the previous
passing model.

Prefer the simpler model whenever performance differences remain inside the
predeclared tolerances. Do not invent a weighted score that can hide a failed
metric.

### Phase 3: targeted interactions

Test only interactions with a concrete redundancy hypothesis:

- Sinkhorn versus dead-topic recycling.
- Positive-NPMI versus prototype separation.
- Additive document evidence versus the multiplicative document gate.
- Paired views versus Jensen-Shannon consistency and local reconstruction.
- Mean-normalized channel evidence versus the equal-channel pull.

An individually failed removal may pass when a redundant partner is retained
or simplified differently. Avoid an exhaustive factorial grid.

### Phase 4: lock and confirm the minimal model

When one cumulative candidate passes all gates:

1. Freeze its architecture and validation decision before reading candidate
   test outputs.
2. Retrain it once in a fresh run directory and confirm validation stability.
3. Rewrite the implementation as that one unconditional model. Delete all
   toggles, rejected code, temporary outputs, and ablation-only abstractions.
4. Update equations, Methods, results, tests, and this handover together.
5. Perform exactly one test evaluation for the locked candidate.
6. If preparing a publication claim, only then run a small predeclared set of
   additional seeds for both the full and minimal neural models. Seed 42 remains
   the development instrument; other seeds are confirmation, not tuning.

### Ablation ledger

Keep one simple `ablation_results.json` outside the final model package while
the study is active. Each row should contain:

- experiment name and removed component;
- training status and seconds;
- validation coverage, evaluable motifs, useful motifs, mean/median SOS, NLL;
- pass/fail for each retention gate;
- parameter-count or code-path reduction;
- concise failure note.

Do not store model-file hashes, schema versions, duplicate protocol copies, or
full rejected checkpoints. Retain only the accepted control and final candidate
needed to audit the scientific conclusion.

## Definition of completion

The ablation task is complete only when the repository again contains one
unconditional model, one evaluation path, one artifact contract, and one
report source of truth; every retained component has validation evidence that
its removal was harmful or that it is inseparable from a retained interaction;
the minimal model passes the full gate; code and LaTeX express the same model;
all tests and report checks pass; and the work is available in a reviewable PR.

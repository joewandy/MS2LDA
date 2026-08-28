# M1 multi-seed stability handoff

## Objective

Quantify optimization-seed stability of the locked M1 architecture on the exact existing seed-42 MSnLib data split. This campaign is not another architecture search.

The scientific question is:

> When data, vocabulary, SGNS features, protocol, and evaluation are fixed, does M1 preserve its chemically useful motif breadth across independent model initializations and training orders?

## Starting point

Use branch `research/neural-ms2lda-finalize-20260828` after its final commit. Create a new execution branch, for example:

`experiment/m1-multiseed-stability-20260828`

Do not work on `main`, the finalization branch, or any prior experiment branch. Do not merge to `main`.

Read first:

1. `benchmarks/neural_ms2lda/FINAL_MODEL_SELECTION.md`
2. `docs/research/neural_ms2lda_report.tex`
3. `benchmarks/neural_ms2lda/protocol.json`
4. `benchmarks/neural_ms2lda/evaluation.py`
5. `benchmarks/neural_ms2lda/diagnostics.py`
6. the locked M1 implementation and simplification history.

## Evidence boundary

Validation only. Do not load, inspect, score, or summarize candidate test theta, completion, MAG/SOS, or result artifacts.

The existing seed-42 test result remains reporting-only evidence for the already locked model. New optimization seeds must not be evaluated on test during this campaign.

## Isolate optimization variation from data variation

Keep all of the following byte-identical across runs:

- seed-42 scaffold/compound split;
- prepared train and validation matrices;
- validation records;
- train-only vocabulary and its order;
- train-only SGNS/token feature artifact;
- positive-NPMI graph;
- MAG leakage-filtered reference index;
- model architecture and every non-seed hyperparameter;
- completion, MAG, SOS, and diagnostic code.

Vary only the model/training seed used for:

- prototype initialization;
- learned parameter initialization;
- PyTorch RNG;
- mini-batch/permutation order;
- any other stochastic operation inside model fitting.

Do not change the data split seed. The current repository uses one protocol seed in several places, so implement an explicit `training_seed` override rather than editing `protocol.json` and accidentally rebuilding the split or SGNS features.

Record both:

- `data_seed = 42`
- `training_seed = <seed>`

in every artifact and result.

## Predeclared seeds

Run new training seeds:

- 11
- 23
- 37

Use the existing locked seed-42 run as the reference fourth seed. Do not choose replacement seeds after seeing results.

Before the three new full runs, replay seed 42 once through the new training-seed plumbing and verify that it reproduces the existing locked state and validation outputs exactly or to the repository's already established deterministic contract. If it does not, stop and repair the seed plumbing before continuing.

## Required implementation

Add a narrowly scoped training-seed option to the M1 campaign/CLI. It must not alter data preparation or feature construction.

Recommended artifact layout:

`<campaign-root>/training_seed_<seed>/`

Reuse or link immutable prepared artifacts rather than copying multi-GB inputs. Fail closed if hashes of the prepared train matrix, validation matrices, vocabulary, token features, co-occurrence graph, protocol, or validation records differ across seeds.

Each run must save:

- exact training seed;
- source commit and dirty-state check;
- hashes of all shared immutable artifacts;
- trained-model state hash;
- training history;
- validation beta and theta hashes;
- validation completion metrics;
- validation MAG/SOS metrics;
- full model-selection diagnostics from `diagnostics.py`;
- fitting/inference time and peak memory;
- finite/stability status.

## Frozen architecture and protocol

Do not tune M1 between seeds.

Use exactly:

- K=1000;
- 48D SGNS + two channel indicators;
- 128D learned projection;
- current nonlinear leave-one-out router;
- top-2 routing;
- additive document score and gate exponent 0.75;
- fixed 50/50 fragment/loss decoder;
- current alternating router/topic schedule;
- current Sinkhorn, NPMI, separation, annealing, optimizer, batch sizes, and epoch count.

Do not add early stopping, change epochs, or rescue a weak seed with a different hyperparameter.

## Validation metrics

For every seed report:

### Chemistry

- optimized motifs;
- evaluable motifs;
- useful motifs;
- mean and median SOS;
- SOS bands;
- associated spectra and molecules;
- leakage-audit status.

### Completion

- validation NLL per in-vocabulary token;
- OOV fraction.

### Inventory and collapse

- median and mean effective topics per spectrum;
- corpus effective topic count;
- active topics above 0.0005 usage;
- active topics at least 1/K usage;
- maximum mean topic usage;
- unique top-1 topics and topics never top-1;
- mean/median nearest beta cosine;
- maximum pairwise beta cosine;
- duplicate-component summaries at 0.95, 0.99, and 0.999;
- median beta effective words;
- median beta maximum probability;
- median beta top-20 mass;
- top-word uniqueness;
- fragment/loss mass invariant.

## Frozen per-seed gates

Use the same chemistry-first gates applied during simplification:

- optimized motifs >= 840;
- evaluable motifs >= 388;
- useful motifs >= 252;
- mean SOS >= 0.651498;
- completion NLL <= 9.422847;
- finite/stable execution;
- no catastrophic duplicate component under the repository diagnostic contract.

Do not weaken these after observing seed results.

## Stability summary

Produce both individual-seed and aggregate summaries.

For each numeric metric report:

- mean;
- standard deviation;
- median;
- minimum and maximum;
- coefficient of variation where meaningful.

Also report pairwise topic alignment between seeds. Because topic labels are exchangeable, do not compare topic indices directly. Match beta rows with a one-to-one Hungarian assignment using cosine similarity, then report:

- mean and median matched beta cosine;
- lower percentiles;
- fraction of matched topics above 0.8, 0.9, and 0.95;
- overlap of useful/evaluable topic inventories after matching where definable.

This cross-seed alignment is descriptive. The primary stability claim remains the chemistry-first validation metrics.

## Predeclared campaign decision

Classify the result as:

### Stable

All three new seeds pass every frozen gate, no seed shows catastrophic collapse, and the aggregate useful/evaluable counts do not show a large unstable tail.

### Mostly stable with one weak seed

Exactly one new seed misses one narrow gate while preserving chemistry breadth and showing no collapse. Report honestly; do not tune. A confirmatory fifth seed may be proposed only after independent review.

### Unstable

Any seed shows catastrophic collapse, multiple chemistry gate failures, or large variation that makes the seed-42 headline unrepresentative.

Do not evaluate test under any classification.

## Result output

Create:

`research/etm_ecrtm_msnlib/local_results/m1_multiseed_<run_id>/`

Commit small reviewable files:

- `README.md`
- `EXPERIMENT_LOG.md`
- `comparison_by_seed.csv`
- `aggregate_summary.csv`
- `pairwise_topic_alignment.csv`
- `provenance.json`
- `environment.txt`
- exact protocol/config copies;
- per-seed metrics and training-history CSVs;
- per-seed diagnostic summaries;
- motif-level chemical score tables where practical.

Do not commit model weights, beta/theta matrices, FAISS indexes, databases, or downloaded data. Record paths, sizes, and SHA-256 values for important local artifacts.

## Quality checks

Before pushing:

- run the full neural benchmark tests;
- run Black and Ruff;
- verify all shared prepared-artifact hashes are identical;
- verify no candidate test artifact was accessed;
- verify seed-42 replay determinism;
- verify CSV/JSON consistency;
- inspect staged files for large binaries.

Push the result branch without merging or opening a PR unless explicitly requested.

The final response must provide the exact branch, final commit, compact per-seed table, aggregate variability, gate decisions, topic-alignment summary, and large-artifact hashes.

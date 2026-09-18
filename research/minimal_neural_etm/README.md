# Minimal neural MS2LDA experiment

This is a development study on `experiment/minimal-neural-topic-model`.
The existing Contextual Sparse ETM and its sealed evidence are unchanged.

**Subsequent independent review (7 September):**
[The expanded review](review_20260907/README.md) now tests the previously
proposed BN-entmax combination, removal of channel balancing, Pyro-reference
ProdLDA, direct-Dirichlet neural models, and ProdLDA plus entmax. It adds 59
training runs and updates the canonical scientific report. The original
round below is preserved as historical evidence, including its then-untested
follow-up suggestions; those suggestions are no longer the latest status.

## Decision from this round

**Keep the current model; retain fixed-scale BatchNorm ETM as a simpler research
baseline, not a replacement.** One published normalization addition substantially
improves useful-motif coverage over plain ETM, but it does not preserve compact
spectral mixtures. Adding the tested sparse prior does not solve that trade-off.

All new candidates below use final **training-only normalization statistics**.
References come from the existing
[validation comparison](../contextual_sparse_etm_msnlib/evidence/20260901_clean_room/validation_comparison.csv),
not its test results. All neural runs use K=1,000, the same input files, initialization
seed, shuffle seed, learning rate, Adam beta1=0.9 and 120 epochs.
Tomotopy retains its separate reference training protocol.

| Model | Evaluable motifs | Useful motifs | Mean SoS | Completion NLL | Median effective topics | Winning topics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain ETM, reference | 199 | 118 | 0.635 | 8.677 | 40.46 | 272 |
| Channel-balanced ETM, reference | 228 | 153 | 0.647 | 8.775 | 45.09 | 254 |
| ETM + learned-scale BatchNorm | 201 | 123 | 0.627 | 8.757 | 39.56 | 288 |
| ETM + learned-scale BatchNorm + prior | 47 | 26 | 0.626 | 9.025 | 2.96 | 56 |
| ETM + fixed-scale BatchNorm | 459 | 296 | 0.651 | 8.231 | 635.33 | 568 |
| ETM + fixed-scale BatchNorm + prior | 234 | 156 | 0.656 | 8.781 | 249.25 | 256 |
| Current Contextual Sparse ETM, reference | 672 | 400 | 0.636 | 9.515 | 3.66 | 838 |
| Tomotopy LDA, reference | 573 | 359 | 0.642 | 9.689 | — | — |

“Useful” is the existing chemical benchmark's operational definition, not a
ground-truth count of distinct substructures. Effective topics measures the
entropy of each spectrum's mixture; winning topics counts distinct dominant
topics across validation spectra. These are different from similarity between
the learned motif distributions.

The fixed-scale, no-prior model finds 2.51 times as many useful motifs as plain
ETM, but only 74% of the current model's count. Its better predictive NLL does
not make its very diffuse mixture assignments suitable as a drop-in MS2LDA
replacement. The learned-scale prior model is sparse but uses very few topics;
the fixed-scale prior model is both diffuse and highly repetitive (mean nearest
beta cosine 0.972). No tested prior setting is recommended here.

The next focused hypothesis would be **ETM + fixed-scale BatchNorm + entmax**:
retain the more active neural encoder and use the published
[sparse transform](https://aclanthology.org/P19-1146/)
already present in the current model instead of relying on this prior for
compact mixtures. That combination has **not** been implemented or evaluated
in this round and is not a demonstrated solution. If an intact published
anti-collapse architecture is preferred, [ECRTM (ICML 2023)](https://proceedings.mlr.press/v202/wu23c.html)
is another candidate; its stock decoder differs from ETM's additive mixture, so
it needs its own faithful likelihood evaluation rather than relabelling this
experiment as ECRTM.

This screen contains 40 synthetic training runs and four real-data training
runs, plus nine paired statistics-recalculation evaluations. All weak results
are retained in [the compact evidence](evidence/runs.csv). Real-data results
use one training seed and one existing validation split; they do not establish
cross-seed robustness or performance on an independent dataset.

## Question and model lineage

Can ordinary ETM recover and use spectral motifs with no more than two changes?

- Published base: Dieng, Ruiz and Blei, [Topic Modeling in Embedding Spaces](https://aclanthology.org/2020.tacl-1.29/), TACL 2020.
- Addition A: batch normalization of the encoder's mean and log-variance heads.
  Both learned-scale and fixed-unit-scale versions are evaluated. AVITM's
  [original implementation](https://github.com/akashgit/autoencoding_vi_for_topic_models/blob/master/models/nvlda.py)
  uses TensorFlow's `scale=False` default; the
  [published ECRTM implementation](https://github.com/bobxwu/ECRTM/blob/d7cc442a526522357f57f3df26ca8f20c485370d/ECRTM/models/ECRTM.py)
  also freezes the two scales. Biases remain trainable.
- Addition B: a diagonal logistic-normal approximation to a symmetric Dirichlet
  prior. For per-topic concentration `a`, the logit variance is `(1 - 1/K)/a`.
- Both additions are motivated by Srivastava and Sutton,
  [Autoencoding Variational Inference for Topic Models](https://arxiv.org/abs/1703.01488),
  ICLR 2017, equations 6–7 and section 3.4.

This is ETM with selected published inference choices, **not** an exact
reproduction of AVITM/ProdLDA. It retains ETM's two-layer ReLU encoder, fixed
training-only SGNS embeddings, softmax mixture, and additive multinomial
decoder. Neither candidate uses contextual routing, entmax, or fragment/loss
balancing. The Dirichlet approximation does not create exact zeros.

Implementation: `benchmarks/neural_ms2lda/prior_etm.py`.
Runner: `scripts/run_minimal_etm.py`.

## Development protocol

The first exploratory screen used synthetic seed 11. The follow-up comparison
uses seeds 11, 23 and 37, with 18 planted motifs and 36 or 128 fitted topics.
The experiment is a validation development study, not a preregistered test.

The same data, train-only SGNS vectors, 800-unit encoder, 120 epochs, batch
size 200, raw intensity pseudo-counts, Adam learning rate 0.005, and weight decay
1.2e-6 are used for the candidates and rerun controls. Adam beta1 is 0.99,
following AVITM's high-momentum recommendation; historical study results used
0.9 and are not interchangeable with these rerun controls. A matched 0.9
comparison checks sensitivity to this optimization choice.

Variants are ordinary ETM, balanced ETM, the current Contextual Sparse ETM,
ETM + BatchNorm, ETM + sparse prior, and ETM + both. Sparse-prior concentration
is 0.02 per topic. This is a fixed pilot setting, not a claim that total prior
concentration is constant as K changes.

After the initial screen, fixed-scale BatchNorm was added as a normalization
recipe comparison, not as a third model ingredient. A train-only recalculation
of the final normalization statistics was subsequently added after diagnosing
severe lag in the running statistics. These are sequential development choices,
not a preregistered search.

Model fitting receives only the training matrix. Synthetic truth and validation
spectra are used after fitting. Real-data runs accept only the repository's
sealed train/validation view. The original test outcomes are already known;
no new test evaluation or model selection on test matrices is performed.

Preference is for the fewest additions that retain strong motif recovery,
compact mixtures, and a broadly used inventory. A high winner count by itself
is not success: duplicated or chemically unhelpful motifs must also be checked.
Real-data chemical evaluation is needed before claiming that the current model
can be replaced. Seed robustness on this split is not external validation.

## Reproduce a synthetic run

Run in the repository's `ms2lda-neural` Conda environment:

```bash
python -m scripts.prepare_contextual_sparse_etm_synthetic \
  --output-root output/benchmarks/minimal_neural_20260905 --seed 11
python -m scripts.run_minimal_etm \
  --synthetic-root output/benchmarks/minimal_neural_20260905 \
  --output output/benchmarks/minimal_neural_20260905/example \
  --variant batchnorm --topics 36 --seed 11 --normalization-statistics ema
```

Every output directory must be new. Each run saves configuration, exact input
and implementation hashes, a model checkpoint (including BatchNorm statistics),
training history, probability matrices, validation metrics and synthetic truth
recovery where available. Large run artifacts and public input downloads remain
under the locally excluded `output/benchmarks/` directory.

The command above reproduces the initial EMA-statistics screen. The runner now
defaults to `--normalization-statistics training` for reliable frozen inference;
this recomputes both posterior heads' mean and sample variance over **training
spectra only** at the final encoder weights. It does not fit on validation data,
change learned weights, or introduce another loss term. Welford accumulation
in float64 avoids cancellation in heads with large offsets and small variance.

The independent tests compare the KL to PyTorch distributions, verify the
unmodified ETM limit, and check that validation inference is independent of
batch composition and does not update BatchNorm statistics.

## Synthetic findings

The initial EMA-statistics beta1=0.99 comparison across seeds 11, 23 and 37 gives the following
means. Beta/theta recovery are the repository's truth-matched cosine metrics;
recovered counts motifs with matched beta cosine at least 0.50. There are 18
planted motifs. These development results motivate real-data validation.

| Fitted topics | Model | Beta recovery | Theta recovery | Recovered motifs | Completion NLL | Median effective topics |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 36 | ETM | 0.382 | 0.643 | 5.33 | 6.451 | 2.76 |
| 36 | ETM + BatchNorm | 0.575 | 0.861 | 12.67 | 6.145 | 3.13 |
| 36 | ETM + prior + BatchNorm | 0.590 | 0.886 | 13.33 | 6.182 | 2.14 |
| 36 | Contextual Sparse ETM | 0.569 | 0.850 | 12.67 | 6.202 | 1.91 |
| 128 | ETM | 0.408 | 0.589 | 6.33 | 6.441 | 3.51 |
| 128 | ETM + BatchNorm | 0.651 | 0.947 | 15.67 | 6.153 | 3.32 |
| 128 | ETM + prior + BatchNorm | 0.651 | 0.951 | 15.67 | 6.194 | 2.63 |
| 128 | Contextual Sparse ETM | 0.659 | 0.946 | 16.00 | 6.178 | 1.73 |

The prior alone was weak in the seed-11 pilot (one motif recovered, three
winning topics), and is retained in the evidence rather than omitted. BatchNorm
alone was the preferred candidate **for advancing to real-data validation**;
the prior produces more concentrated mixtures but does not consistently improve
synthetic recovery or predictive fit. This is not a replacement recommendation.

The matched seed-11 check at the original beta1=0.9 also supports BatchNorm:
at K=128 it recovers 16 motifs, compared with plain ETM's 5 and the current
model's 18; its theta recovery is 0.960 and NLL 6.066. At K=36 it recovers 11,
compared with plain ETM's 3 and the current model's 10. This check is one seed,
not a second three-seed study.

The initial BatchNorm runs used learned affine parameters and frozen exponential
moving-average statistics. The generator remains ETM, and softmax still gives dense mathematical support.
BatchNorm does not guarantee that all topics will be useful or eliminate
collapse on every dataset. Synthetic recovery, winner counts, duplication and
real chemical usefulness answer different questions and should be read together.

### Fixed scales and the inference-statistics check

At K=128 and beta1=0.9, all three fixed-scale, no-prior runs initially assigned
every validation spectrum to a single dominant topic. This was an inference
statistics failure, not evidence that their learned motif matrices contained
only one motif. In seed 11 the largest posterior-mean running-statistic error
was 18.52, while the median actual head variance was only about 0.00073.
Recomputing statistics from training spectra, without retraining, restored
30 winning topics and theta recovery of 0.938 for that run.

The paired **training-calibrated** means over seeds 11, 23 and 37 are:

| Model, K=128 | Beta recovery | Theta recovery | Recovered motifs | Completion NLL | Median effective topics | Winning topics |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| ETM + fixed-scale BatchNorm | 0.655 | 0.889 | 16.67 | 6.110 | 23.60 | 29.33 |
| ETM + fixed-scale BatchNorm + prior | 0.642 | 0.949 | 15.00 | 6.236 | 19.46 | 33.33 |

The recalculated results are saved separately in `fixed_scale_calibrated/`;
the original `fixed_scale/` outputs remain intact. Both versions still have
substantially more diffuse mixtures than the current Contextual Sparse ETM.

## Real-data validation

Use the public assets and the existing split/index preparation, then seal a
separate view for each candidate. Real runs use 1,000 topics, seed 7043
(`--seed 42` in this runner), 120 epochs and the original Adam beta1=0.9.

```bash
python -m scripts.download_msnlib_validation_assets --data-root RUN/assets
python -m scripts.prepare_contextual_sparse_etm_data \
  --run RUN/prepared --data-root RUN/assets
python -m benchmarks.neural_ms2lda.mag \
  --run RUN/prepared --data-root RUN/assets
python -m scripts.prepare_msnlib_validation_view \
  --run RUN/real/batchnorm --prepared-run RUN/prepared
python -m scripts.run_minimal_etm \
  --validation-run RUN/real/batchnorm \
  --output RUN/real/batchnorm/models/minimal_etm \
  --variant batchnorm --topics 1000 --seed 42 --momentum 0.9
python -m benchmarks.neural_ms2lda.chemical \
  --run RUN/real/batchnorm --data-root RUN/assets \
  --method minimal_etm --split validation
```

The `minimal_etm` chemistry entry rejects test evaluation. Preparing the shared
split and filtering the reference index necessarily handles held-out record
identifiers; model fitting and validation do not load test matrices. The new
training/validation matrices, records, vocabulary, SGNS features and filtered
MAG index match the September 1 study byte-for-byte. The protocols are equal
as JSON objects (their whitespace formatting differs).

Compact evidence is generated by:

```bash
python -m scripts.summarize_minimal_etm \
  --root output/benchmarks/minimal_neural_20260905 \
  --output research/minimal_neural_etm/evidence
```

To correct an existing candidate's statistics without repeating training, first
create a fresh sealed validation view and run:

```bash
python -m scripts.run_minimal_etm \
  --validation-run RUN/real_calibrated/batchnorm \
  --output RUN/real_calibrated/batchnorm/models/minimal_etm \
  --variant batchnorm \
  --recalibrate-from RUN/real/batchnorm/models/minimal_etm
```

The source's training recipe and input hashes are checked and retained. Parent
checkpoint/result hashes distinguish these derived evaluations from independent
training runs. `reused_training_weights` in the CSV prevents counting a
recalibration as a new seed. Original training time is carried forward, not
reported as the cost of recalibration. Early pilot checkpoints contained a
PyTorch `TorchVersion` metadata value; the recalibration reader permits only
that specific safe string subclass. All new checkpoints use ordinary strings
and load with `weights_only=True` without allowlisting.

## Verification and boundaries

- 93 focused neural-model tests pass, including independent KL checks,
  checkpoint round-trips, train-only calibration, preserved weights on
  recalibration, and rejection of exposed test inputs/test chemical evaluation.
- Exact CI Black/Ruff scopes pass. The public wheel builds and imports from a
  separate installation target, with the experiment excluded from the wheel.
- `pip check` passes in `ms2lda-neural`; the existing study's LaTeX report also
  compiles. Its source and frozen evidence were not changed.
- Calibration records and input/code hashes are stored beside each new model.
  The compact evidence summary rejects held-out-reference contamination or
  unresolved chemical annotation failures.
- The fixed-scale no-prior checkpoint reloads bit-identically on GPU for all
  3,889 validation mixtures. CPU inference is close, not bit-identical: its
  maximum probability difference is 0.0000843 and four spectra change dominant
  topic. These cross-device differences are recorded in
  [the checkpoint audit](evidence/checkpoint_inference.json); the reported
  chemical scores use the saved GPU predictions.
- This is a model-form experiment, not parameter compression: the 800-unit
  encoder still dominates approximately 19.3 million stored parameters at
  K=1,000. No inference speed claim is inferred from these training timings,
  some of which involved GPU contention.
- No production integration, test-set evaluation, commit, or push is performed
  as part of this development screen. The unrelated local Spec2Vec artifacts
  are preserved.

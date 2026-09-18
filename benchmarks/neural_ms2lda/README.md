# Neural topic models for mass spectra

This is research code, separate from the production Tomotopy backend. The
[main paper](../../docs/research/contextual_sparse_etm_report.tex) explains the
base model and three enhancements; its
[supplement](../../docs/research/contextual_sparse_etm_supplement.tex) gives the
complete equations, evaluation protocols and reproducibility details. Both
describe one chosen model: `ReducedContextualETM` with
`variant="reduced_document_context"`. Its retained two-hidden-layer MLP and embedded
decoder come from the published Embedded Topic Model (ETM).

Read the selected model as ordinary tensor functions, not as an extensible
framework. Small PyTorch modules own parameters and checkpoint state; data
loading, inference, losses, metrics and evidence checks are functions. The
remaining variant switches preserve already-run ablations. They are not
additional components of the selected model.

## From the paper to the code

For a batch of `B` spectra, `V` words, `K` topics and embedding dimension `E`,
`c` is the `B × V` pseudo-count matrix, `x` its row-normalized encoder input,
`rho` the fixed `V × E` word embeddings, and `alpha` the learned `K × E` topic
embeddings. The paper indexes spectra, words and topics by `d`, `w` and `k`.
Fragments and neutral losses are separate words even at the same m/z.

The published backbone uses a Gaussian encoder
`q(z_d | x_d) = Normal(m_d, diag(exp(ell_d)))` and a standard-normal prior.
Here `ell` is **log variance**, not log standard deviation. The decoder gives
`p(w | z_d) = sum_k theta_dk beta_kw`. The chosen model adds only:

1. **Balanced emissions:** separate softmax normalization within fragments and
   losses, each with mass one half; each topic still sums to one over words.
2. **Whole-spectrum evidence:** the count-weighted mean word direction modifies
   each observed word direction through one learned scalar. Each word assigns
   cosine-based softmax evidence to its top two topics. Count-weighted pooling
   gives `r_d`; `center(log(r_d + 1/K))` shifts the MLP posterior mean.
3. **Sparse mixtures:** 1.5-entmax maps a reparameterized Gaussian sample to
   `theta_d` during training. Inference uses entmax of the posterior mean;
   this is a deterministic summary, not the expected mixture under `q`.

The scored word **is included** in the whole-spectrum context. Top two is a
per-word choice, not a two-topic limit for a spectrum. Context evidence is not
an extra likelihood or a claimed exact word-topic posterior. The shifted mean
enters both reconstruction and Gaussian KL; there is no auxiliary loss.

| Paper operation | Canonical implementation |
| --- | --- |
| Normalize encoder input; keep raw likelihood targets | `topic_model_training.dense_normalized` |
| Published embedded decoder (`eq:base-beta`) and Gaussian MLP | `etm_baselines.CanonicalETM` |
| Balanced decoder (`eq:beta`) | `contextual_sparse_etm.channel_balanced_topic_word_distribution` |
| Full-spectrum context and evidence (`eq:document-context`, `eq:document-evidence`) | `contextual_reductions.pooled_evidence(context="document", routing=2)` |
| Mean offset (`eq:posterior-offset`) | `contextual_sparse_etm.centered_log_evidence_offset` |
| Complete posterior, including its shifted-mean KL | `contextual_reductions.ReducedContextualETM.posterior` |
| Gaussian sampling, KL and sparse mixture | `contextual_sparse_etm.reparameterized_gaussian`, `diagonal_gaussian_kl`, `entmax15_document_mixture` |
| Negative reconstruction term (`eq:elbo`) | `topic_model_training.sparse_reconstruction_loss(scaling="raw_counts")` |
| Mean reconstruction + mean KL, coefficient one | `scripts/run_minimal_etm.fit` |
| Batched deterministic inference | `model_inference.infer_document_topics` |
| Synthetic matching and recovery | `synthetic_msms.matched_truth_metrics` |
| MAG annotation and SOS evaluation | `mag.py`, `chemical.py` |

### Numerical and evaluation conventions

These details qualify the equations; they are not extra model components.

- Define `x = c / N` for positive in-vocabulary mass and `x = 0` otherwise.
  An empty observed completion half gets uniform contextual evidence (zero mean
  offset) and the MLP's learned zero-input posterior. It remains scoreable when
  its withheld half has in-vocabulary counts. Tomotopy's corresponding fallback
  is its learned Dirichlet prior normalized over topics.
- The ideal model uses the multinomial log likelihood. The executable training
  and completion calculations use `log(max(p, 1e-12))`; below the floor the
  reconstruction gradient with respect to `p` is zero. Entropy diagnostics also
  floor positive probabilities inside their logarithm. The fitting recipe's
  Adam weight decay is an optimizer regularizer, not an additional contextual
  or motif-quality objective.
- Real completion re-max-normalizes observed peak intensities and re-discretizes
  them. Synthetic completion preserves the generated integer counts when
  splitting physical peak groups; the encoder subsequently normalizes its row.
  These are intentionally documented as different historical preparations.
- Synthetic matching requires enough fitted topics to match every planted
  motif (`K >= R`) and aligned, finite nonnegative arrays with positive row
  mass. An inferred mixture can still have zero mass on the **matched subset**;
  that spectrum then receives zero mixture-recovery cosine.
- MAG's legacy `cosine_similarity=0.9` control produces complete-linkage
  Euclidean clustering of masked-motif similarity profiles at distance `0.1`.
  It is not a direct pairwise spectral-cosine cutoff. SOS requires available
  consensus fingerprints; an all-zero consensus scores zero, while an
  unavailable consensus makes a motif unevaluable.

Equation labels in this table and the code refer to the
[supplement's detailed formulation](../../docs/research/contextual_sparse_etm_supplement.tex).
The main paper retains the base ETM equations and links to the relevant
supplementary sections. Corresponding docstrings explain tensor shapes,
normalization axes and numerical safeguards.
`tests/test_contextual_reductions.py` independently constructs the selected
equations; `tests/test_review_boundaries.py` also checks the analytic derivative
of the centered log offset at uniform evidence. The complete selected-model
reference calculation and preparation/metric boundary tests are in
`tests/test_current_model_equations.py`. They use only small CPU examples,
without fitting models or loading reserved test data. The
[equation audit](../../docs/research/equation_code_audit_20260908.md) records the
full numbered/unnumbered correspondence and checks on saved validation evidence.
The inherited MAG clustering check lives separately in the production suite,
`tests/test_mag_profile_clustering.py` relative to the repository root. This
keeps the focused research tests independent of MAG's full dependency stack;
the production CI job runs that check without optional skips.

## Data, execution and evidence

`data.py` and `spectra.py` prepare connectivity-group splits and spectral words.
`validation_data.load_validation_inputs` checks the sealed train/validation
view, file bytes and aligned shapes before fitting. Vocabulary and SGNS
embeddings are trained on training spectra only. No test matrices are allowed
in a fitting view. Frozen-test release checks live separately in `test_release.py`.

`scripts/run_minimal_etm.py` constructs a declared variant, fits the training
matrix, and records input/source identities, recipe and recovery checkpoints.
Exact trajectory recovery requires matching source, inputs, runtime and
hardware; a checkpoint alone does not promise cross-version reproducibility.

The chosen model's nine fits and selection reasoning are preserved in
[the research review](../../research/minimal_neural_etm/review_20260907/README.md).
Its report generator is `scripts/generate_contextual_reduction_review.py`.
Tomotopy is the primary comparator and plain ETM the secondary enhancement
reference. These are development-validation results, not an independent
post-selection test.

The fixed comparator repetitions use `scripts/run_baseline_repeats.py`: its
`--root` must be a fresh output directory, `--prepared-run` supplies the frozen
training/validation inputs and SGNS embeddings, and `--data-root` supplies the
annotation assets. The runner fits three one-worker Tomotopy models and two
plain ETM models; the original plain ETM fit is retained as its third repeat.
After all stages succeed, `python -m
benchmarks.neural_ms2lda.baseline_repeat_evidence --root RUN_ROOT --output
EVIDENCE_DIRECTORY` validates and packages the six-fit inventory. Use a fresh
evidence destination; existing results are not overwritten. The final report
uses `research/baseline_repeats_20260908/evidence/`. Its manifest includes 21
execution/cache-recheck logs as well as separate fitting/postprocessing source
archives; all manifest-listed files must accompany a shared copy.

The older `ContextualSparseETM` class uses leave-one-out context. Its frozen
clean-room evidence remains under
`research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room/`.
`study_protocol.py`, `reproduction_plan.py`, and the
`run_contextual_sparse_etm_reproduction` CLI describe that historical study,
**not** a rerun of the selected whole-spectrum model. Comparison and summary
functions live in `reproduction_comparison.py` and `reproduction_summaries.py`;
the packager only verifies and publishes their outputs atomically.

The 8 September code review corrected the uniform-evidence offset's gradient
while preserving its exact-zero forward value. Saved fits/results were not
retrained or replaced. Use their recorded source snapshots for exact historical
replay. See the [review record](../../docs/research/code_quality_review_20260908.md)
for this distinction and the bounded before/after numerical checks. The later
equation audit proves that this uniform-only branch is unreachable for all
three chosen real fits and all three chosen synthetic `K=128` fits: each
training spectrum has fewer than `K/2` observed word types, and top-two evidence
therefore cannot occupy every topic. The `K=36` trajectories retain the stated
historical-source caveat; no change to their recorded outputs is implied.

## Verification

Run from the repository root in the canonical `ms2lda-neural` environment:

```bash
pytest -q benchmarks/neural_ms2lda/tests
NUMBA_DISABLE_JIT=1 pytest -q tests
python -S -m scripts.generate_contextual_reduction_review
python -S -m scripts.generate_motif_chemical_assessment_report --tables-only
python -S -m scripts.generate_cross_model_overlap_report --tables-only
```

These three generators produce the current manuscript's model comparison,
chemical assessment and cross-model correspondence inputs, respectively.
The exact Black/Ruff research scope, all six report generators, production
regressions and isolated-wheel checks are declared in
[the research workflow](../../.github/workflows/neural-ms2lda.yml). The other
three generators, `generate_contextual_sparse_etm_report`,
`generate_minimal_etm_review` and `generate_published_neural_review`, maintain
archival review tables; those tables do not enter the current main paper or
supplement.

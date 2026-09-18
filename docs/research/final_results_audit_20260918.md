# Final results audit, 18 September 2026

The displayed results are consistent with the frozen evidence. All current
numerical report fragments and all five empirical figures regenerated without
a byte change. Reloading all nine selected-model checkpoints reproduced their
saved topic probabilities and full-spectrum mixtures exactly; independently
calculated completion scores agree within floating-point rounding. No model
was retrained, no test partition was evaluated, and no scientific evidence was
rewritten in this audit.

The one additional reporting issue found was the substantial out-of-vocabulary
fraction in synthetic completion. The supplement now states its range and
distinguishes vocabulary-restricted word recovery from full-signal mixture
truth. This disclosure changes no result or ranking.

## Scope and input identities

The audit covered the main manuscript and supplement, their current generated
tables/macros, five empirical figures, and these evidence collections:

- `research/minimal_neural_etm/review_20260907/evidence/within_model/`
- `research/baseline_repeats_20260908/evidence/`
- `research/motif_chemical_assessment_20260908/evidence/`
- `research/cross_model_overlap_20260908/evidence/`

The baseline reader verified its complete public hash manifest and the six
declared fit identities: Tomotopy seeds 11, 23 and 42; plain ETM training seeds
7012, 7024 and 7043. All 127 local sources listed in the within-model manifest
were independently SHA-256 checked, as were all 51 input paths consumed by the
nine selected fits. Selected real runs have labels 11, 23 and 42 and training
seeds 7012, 7024 and 7043. Synthetic runs have realization seeds 11, 23 and 37,
training seeds 7012, 7024 and 7038, and both K=36 and K=128.

Selected checkpoint paths are beneath
`output/benchmarks/contextual_reduction_20260907/`:

| Fit | Checkpoint SHA-256 |
| --- | --- |
| `real/reduced_document_context_seed11/models/minimal_etm/checkpoint.pt` | `96e2b9063efca8fa7891c3837056121ea5a600b93e0e122414a3c57d4207ae76` |
| `real/reduced_document_context_seed23/models/minimal_etm/checkpoint.pt` | `41de7fc3c88d67fbd33c403a12e2f6d23c1537970cd622acf4ace1b5997a1f8f` |
| `real/reduced_document_context_seed42_retry1/models/minimal_etm/checkpoint.pt` | `d561a122991c9f29e13c30b891d76f401a7dcd554f9acaeeda842d831faafe8c` |
| `synthetic/reduced_document_context_seed11_k128/checkpoint.pt` | `2fb1ec585e9e68485c4f7671c9d26e32676a14b41876eac551dfcd5a2eafbe4d` |
| `synthetic/reduced_document_context_seed11_k36/checkpoint.pt` | `cadc4d231097d1f801b85b0eae2a8e943de07f7d4d06945e1f2512d5d62f72eb` |
| `synthetic/reduced_document_context_seed23_k128/checkpoint.pt` | `773459b9d5ff7cfc91bd4cd0d8bb06d50d343903bae16cb1d9051d1395481e05` |
| `synthetic/reduced_document_context_seed23_k36/checkpoint.pt` | `78f29e72eeb683d42a4425ef40053a7feacbb814a0dac73a1fe0b91d7e0fb2d2` |
| `synthetic/reduced_document_context_seed37_k128/checkpoint.pt` | `3a58a7a1d44f1c1105e19ad475daf658db8434814f1a8d038e1997ecea42384f` |
| `synthetic/reduced_document_context_seed37_k36/checkpoint.pt` | `dad850a16ca3059e362cf8eff80656bac4e6059ed6412fd234859750137a6599` |

## Independent numerical checks

**Primary and synthetic results.** `current_model_report` reproduced the four
current tables/macros from the selected nine rows and the verified baseline
inventory. Together with chemical and overlap generation, all 11 current
generated text/JSON files were byte-identical. Means use three fit-level
values, SDs use the sample denominator, and effective counts are within-fit
medians before aggregation. In particular, useful motifs remain 390.7 +/- 2.1
for the selected model and 351.7 +/- 7.0 for Tomotopy.

All nine checkpoints were loaded with `weights_only=True`, the current
`build_model` factory and the canonical CUDA environment. No optimizer was
created or run. Exported beta and full validation theta were bitwise identical
to the saved arrays. Only the observed completion matrix entered completion
inference. A separate NumPy float64 calculation summed withheld counts times
`-log(max(theta @ beta, 1e-12))` at nonzero withheld cells and divided by total
withheld in-vocabulary mass. Its largest NLL difference from a saved result
was 4.48e-8. Every real fit scored exactly 1,277,983 tokens and 3,888 eligible
spectra. The real fits had 14, 141 and 28 scored token copies below the stated
probability floor; no synthetic scored token fell below it. The audit verifies
the reported floored metric, not an unfloored likelihood.

Entropy-effective counts, exact support and dominant-topic counts were
independently recomputed from saved mixtures. All selected real fits had
median exact support six; their dominant-topic counts were 817, 799 and 806.
Synthetic recovery was recomputed with float64 row normalization, SciPy's
Hungarian assignment, matched word cosine and aligned mixture cosine. Counts
above the inclusive 0.50 word threshold matched exactly. Differences from
saved float32 word scores were at most 4.82e-7 and do not affect reported
rounding; mixture-score differences were at most 2.32e-9.

For all six baseline fits, per-topic chemical scores independently reproduced
evaluable counts, useful counts and mean SOS, and each NLL matched its sealed
per-fit result. Mixture arrays available for five baseline fits reproduced
dominant-topic counts exactly and effective counts within 1.94e-6, without a
displayed change. Completion inference was subsequently replayed for all five
available baseline models, using only their observed validation halves. The
historical plain-ETM seed 7043 has summary/per-topic evidence but lacks its full
local model arrays, so its mixture and completion inference could not be
replayed.

The two plain-ETM checkpoints used CUDA, their original batch size 256, six
PyTorch threads and deterministic execution with training seeds 7012/7024.
Tomotopy 0.13.0 used a fresh load of each saved model and observed-half
inference as its first sampler call, matching the original evaluation order:
100 iterations, one worker, `parallel=1`, `together=False`, with no additional
seed override, warm-up, full-spectrum inference or training. Three separate
processes ran concurrently, each retaining one inference worker. Every
Tomotopy model hash matched its frozen baseline manifest. The calls completed
in 372.6--376.4 seconds, within the 420-second per-process limit.

| Baseline fit | Saved NLL | Independent float64 replay | Absolute difference |
| --- | ---: | ---: | ---: |
| Tomotopy seed 11 | 9.696947265349634 | 9.696947183815436 | 8.15e-8 |
| Tomotopy seed 23 | 9.692857122517491 | 9.692857040723286 | 8.18e-8 |
| Tomotopy seed 42 | 9.688764501508032 | 9.688764364023283 | 1.37e-7 |
| Plain ETM seed 7012 | 8.711593291272205 | 8.711593176872707 | 1.14e-7 |
| Plain ETM seed 7024 | 8.733827556009679 | 8.733827444357608 | 1.12e-7 |

All five extracted beta arrays were bitwise identical to their saved arrays;
all five completion calculations retained exactly 1,277,983 scored tokens and
3,888 eligible spectra. NLL differences are consistent with float64 versus
the original float32 probability calculations and are far below displayed
precision. A tolerance of 1e-6 covers every baseline replay. Saved observed-
half theta arrays are unavailable, so this does not claim bitwise equality of
the Tomotopy sampler outputs. It is one original-convention replay per fit,
not an estimate of variation across newly chosen inference seeds.

Baseline model SHA-256 identities were:

- Tomotopy 11: `e558624f85049e9a08ec2117bee80792f2ca3086f56048dadf1b68a4f3721283`
- Tomotopy 23: `2e16a41ab88d9377655ffd7dc6a6890336af55da6bf5ca146fd3ac552ac6e033`
- Tomotopy 42: `67b51cb661b61a950f04efa244b9fc8168b963fa20b6b3e5272f64f55e79226a`
- Plain ETM 7012: `516a251431a3147e92791bf5eca79791374beabe889a9eb8beb404d4c86a3df6`
- Plain ETM 7024: `3772748b26df3b381e5ccea3cab6861932198bf07ebf56c53b08ff261bfb27d9`

**Chemical assessment.** The existing full replay was revalidated: 27 payloads
were byte-identical, the three stage summaries were identical, and the two
reference-score payloads differed only by cosine rounding of at most 3.33e-16,
with identities, ranks and overlaps unchanged. This audit did not execute
another 99,999-permutation campaign.

Separate calculations from `compounds.jsonl`, all six motif inventories and
each saved cohort recomputed support, all feature-positive counts, expected
feature counts, observed SOS, matched-background SOS and excess for every
topic in all four configurations. Across 24,000 topic records, maximum SOS
error was 3.33e-16 and maximum expected-feature-count error was 1.99e-13.
The plus-one permutation p-value was checked against stored tail-hit counts
for every informative SOS test. The 12 smallest feature p-values in each
configuration were separately reconstructed by convolving SciPy
hypergeometric probability masses; all 48 agreed, with maximum absolute
error 1.66e-24. The reported bromine and sulfur-in-ring examples are present
in the relevant saved q-value arrays.

SciPy's independent Benjamini--Yekutieli routine reproduced all four complete
families of 996,000 feature tests and 6,000 SOS tests; maximum feature-q
difference was 2.23e-16. Missing/constant tests retain their declared family
slots. Cohorts contain 3,854 compounds with 158/229/106 strata for 50/25/100 Da,
and 2,856 compounds with 124 strata for the scaffold-balanced 50 Da analysis.

**Cross-model recovery.** All four scientific payloads remained byte-identical
to the existing full replay. An independent sorted-score threshold calculation
reproduced every curve for all 133,220 directed rows and 180
fit-pair/cohort/metric groups with zero difference, including every source
denominator. All 133,220 reciprocal flags agreed with reverse matching.
Nearest scores were no smaller than available earlier Hungarian counterparts
in 30,000 all-topic and 9,834 recurring/evaluable comparisons. Aggregation
averages target fits within source fits before the three-source-fit summary;
the nine cross-model pairs are not treated as independent replicates.

**Figures.** Both PDF and PNG outputs for chemical specificity, motif
correspondence, cross-model recovery, cross-model agreement and spectral
examples regenerated byte-identically from sealed evidence. This checks
their plotted data and deterministic rendering; complete manuscript layout
and editorial checks are recorded separately.

## Synthetic vocabulary disclosure

These values come from each selected synthetic `result.json` under the paths
above, at `metrics.completion`, and agree at both fitted topic counts:

| Realization seed | In-vocabulary withheld tokens | Excluded withheld tokens | Excluded fraction |
| --- | ---: | ---: | ---: |
| 11 | 105,704 | 58,030 | 35.4416% |
| 23 | 103,343 | 61,569 | 37.3345% |
| 37 | 106,955 | 58,669 | 35.4230% |

The fraction is excluded/(included+excluded), over pseudo-token copies.
It is not a fraction of spectra or distinct words. In `synthetic_msms.py`,
`_true_beta` uses retained-vocabulary, motif-labeled training counts;
`_true_theta` uses all motif-labeled signal counts before vocabulary filtering,
excluding background/noise. Therefore word recovery concerns retained words,
whereas mixture truth retains signal whose words may be out of vocabulary.
The completion and recovery metrics should not be read as recovery of every
generated spectral feature.

## Commands and execution record

Commands were run in the repository root using the canonical `ms2lda-neural`
environment. The following focused test command passed **140 tests**:

```bash
conda run --no-capture-output -n ms2lda-neural pytest -q \
  benchmarks/neural_ms2lda/tests/test_baseline_repeats.py \
  benchmarks/neural_ms2lda/tests/test_chemical_assessment.py \
  benchmarks/neural_ms2lda/tests/test_cross_model_overlap.py \
  benchmarks/neural_ms2lda/tests/test_report_comparison.py \
  benchmarks/neural_ms2lda/tests/test_contextual_reduction_review.py \
  benchmarks/neural_ms2lda/tests/test_reproduction_evidence.py
```

Both generators were run with separate output paths, followed by SHA-256/byte
comparisons against the checked-in files:

```bash
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 conda run --no-capture-output -n ms2lda-neural \
  python -m scripts.generate_motif_chemical_assessment_report \
  --output /tmp/ms2lda_final_results_audit_20260918 \
  --figures /tmp/ms2lda_final_results_audit_20260918/figures
OPENBLAS_NUM_THREADS=4 OMP_NUM_THREADS=4 conda run --no-capture-output -n ms2lda-neural \
  python -m scripts.generate_cross_model_overlap_report \
  --output /tmp/ms2lda_final_results_audit_20260918 \
  --figures /tmp/ms2lda_final_results_audit_20260918/figures
```

The additional inline Python probes used
`/home/joewandy/miniforge3/envs/ms2lda-neural/bin/python`, with four BLAS/OpenMP
threads for selected-model checkpoint and chemistry calculations. Baseline
completion replay used six BLAS/OpenMP threads and six PyTorch threads for
plain ETM, with Tomotopy's sampler restricted to one worker. The probes called
`read_baseline_evidence`, `current_model_report`, `compare_summaries`,
`compare_payloads`, `check_multiplicity` and the cross-model validator directly;
the protocol-mirror rewriting function was deliberately not called.
Independent recomputation algorithms and their bounds are specified above.
Scratch JSON results are in `/tmp/ms2lda_final_results_audit_20260918/`:
`report_fragment_check.json`, `raw_selected_metrics_check.json`,
`baseline_raw_metrics_check.json`, `checkpoint_nll_replay.json`,
`evidence_validation.json`, `independent_chemistry.json` and
`figure_regeneration_check.json`, `plain_etm_completion_replay.json` and
`tomotopy_completion_seed{11,23,42}.json`. The Tomotopy completion probe was run
for seeds 11, 23 and 42 in separate processes using:

```bash
OPENBLAS_NUM_THREADS=6 OMP_NUM_THREADS=6 PYTHONPATH=/home/joewandy/Work/git/MS2LDA \
  timeout 420 /home/joewandy/miniforge3/envs/ms2lda-neural/bin/python -u \
  /tmp/ms2lda_final_results_audit_20260918/replay_tomotopy_completion.py 11
```

The last argument was changed to 23 and 42 for the other fits. These files and
the probe script are local audit scratch outputs;
the durable evidence remains the committed source bundles and this note.

## Scientific interpretation and limits

The data support the manuscript's descriptive trade-offs: a larger screened
neural inventory and more concentrated mixtures, lower completion NLL than
Tomotopy but higher NLL than plain ETM, and no demonstrated advantage in
background-adjusted chemical specificity. Higher spectral repeatability does
not establish stable supporting-compound assignments or unique substructures.

The review checked fixed-split dependence, conditional SOS populations,
single-compound support, post-selection multiplicity caveats, shared MAG and
MotifDB evidence, target-inventory effects in nearest matching, unequal
optimization budgets and the incomplete final-model ablation design. These
limitations are disclosed; computation does not remove them. Three fits are
not independent chemical populations. Earlier development also used the
reserved test partition, so new independent data are required for confirmation.
Checkpoint replay validates current inference against saved weights; it does
not reproduce full training trajectories, prove optimizer convergence, or
replace independent experimental/chemical validation.

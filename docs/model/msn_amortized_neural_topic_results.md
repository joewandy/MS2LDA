# MSn Amortized Neural Topic Results

## 2026-05-19 2k no-LDA runs

These runs test a neural-only topic model that learns:

```text
theta_local = train-document topic parameters
theta_enc   = encoder(spectrum BoW)
beta        = learned motif-word distributions
```

No tomotopy LDA beta, LDA theta, or `documents.jsonl.gz` are used. Benchmark
export uses full-cache encoder `theta.npy`, learned `beta.npy`, `vocab.json`,
and `split_indices.json`.

Input cache:

```text
notebooks/Paper_results/MSn_evaluation/msn_cache_2000_fixed_beta
```

Runs:

```text
notebooks/Paper_results/MSn_evaluation/amortized_neural_topic_2000_k500
notebooks/Paper_results/MSn_evaluation/amortized_neural_topic_2000_k1000
```

Configuration shared by both runs:

| Setting | Value |
| --- | ---: |
| Spectra | 2,000 |
| Split | 80/10/10 |
| Seed | 42 |
| Epochs | 100 |
| Batch size | 128 |
| Vocabulary size | 3,837 |
| Export theta power | 3.0 |

## Test Split Score

Both models were exported with `--max-eval-motifs 0` and scored on the same
held-out test split.

| Model | Test spectra | Membership rows | Active motifs at 0.5 | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Amortized neural, 500 motifs | 200 | 147 | 119 | 0.238 | 0.5194 | 0.1236 |
| Amortized neural, 1000 motifs | 200 | 143 | 124 | 0.124 | 0.5498 | 0.0682 |

Reference context from earlier all-2k, non-held-out runs:

| Model | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: |
| LDA 2k | 0.592 | 0.5459 | 0.3232 |
| neural-lda 2k | 0.604 | 0.5344 | 0.3228 |

## Interpretation

The no-LDA model did not collapse. The 500-motif run produced 119 active held-out
test motifs and a QAC of `0.1236`, a clear improvement over the fixed-beta
encoder test QAC of `0.0302`. The 1000-motif run produced slightly higher mean
SoS but worse coverage/QAC because the same number of active test motifs is
spread across twice as many total motifs.

The 500-motif run is the stronger setting so far. However, neither no-LDA run is
close enough to the previous 2k LDA/neural-lda reference QAC around `0.323` to
justify a full-cache benchmark yet.

Next experiments should focus on improving held-out coverage before scaling:

- tune encoder/local consistency and encoder topic-usage weights;
- select checkpoints by a validation score that includes active held-out motifs,
  not reconstruction alone;
- try a token-set encoder only after the current dense BoW objective is stable;
- compare held-out train/test scoring against a held-out LDA baseline with the
  same number of motifs.

## 2026-05-20 theta refinement check

`scripts/refine_msn_theta_outputs.py` tests semi-amortized inference without
retraining: start from the neural encoder's `theta_raw.npy`, keep learned
`beta.npy` fixed, run multiplicative KL-NMF-style theta updates against each
observed spectrum, then export the refined theta with the same benchmark
contract.

This still uses no tomotopy LDA beta, theta, or inference. It only uses the
held-out spectrum BoW at inference time, which is the same kind of information
tomotopy held-out inference would use.

| Model | Refine iters | Encoder prior | Test membership rows | Active motifs at 0.5 | Recon before | Recon after | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Amortized neural, 500 motifs | 0 | n/a | 147 | 119 | n/a | n/a | 0.238 | 0.5194 | 0.1236 |
| Amortized neural, 500 motifs | 10 | 0.0 | 93 | 81 | 4.9079 | 4.3495 | 0.162 | 0.5505 | 0.0892 |
| Amortized neural, 500 motifs | 20 | 0.0 | 92 | 80 | 4.9079 | 4.3483 | 0.160 | 0.5539 | 0.0886 |
| Amortized neural, 500 motifs | 10 | 0.1 | 100 | 89 | 4.9079 | 4.3786 | 0.178 | 0.5518 | 0.0982 |
| Amortized neural, 1000 motifs | 0 | n/a | 143 | 124 | n/a | n/a | 0.124 | 0.5498 | 0.0682 |
| Amortized neural, 1000 motifs | 10 | 0.0 | 86 | 82 | 4.5808 | 3.9681 | 0.082 | 0.5281 | 0.0433 |

Refinement reliably improves BoW reconstruction, but it hurts the motif
benchmark. The refined assignments become more reconstruction-optimal under the
learned neural beta while covering fewer useful held-out motifs. A light encoder
prior is the least bad tested setting, but it still stays below the unrefined
500-motif model.

The conclusion is that the remaining issue is not just held-out theta inference.
The learned beta/reconstruction objective itself is not aligned enough with the
MSn motif quality benchmark, so making theta more optimal for that beta can make
the benchmark result worse.

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

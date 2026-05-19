# MSn Fixed-Beta Encoder Results

## 2026-05-19 2k smoke run

This run tests whether a neural BoW encoder can replace tomotopy held-out LDA
inference while preserving the standard benchmark output contract:

```text
theta.npy
beta.npy
vocab.json
annotations.csv
memberships.csv
```

Run directory:

```text
notebooks/Paper_results/MSn_evaluation/fixed_beta_encoder_2000
```

Input cache:

```text
notebooks/Paper_results/MSn_evaluation/msn_cache_2000_fixed_beta
```

Configuration:

| Setting | Value |
| --- | ---: |
| Spectra | 2,000 |
| Split | 80/10/10 |
| Seed | 42 |
| LDA motifs | 1,000 |
| LDA train iterations | 500 |
| Held-out inference iterations | 100 |
| Encoder epochs | 50 |
| Encoder batch size | 128 |
| Cache vocabulary size | 3,837 |
| LDA vocabulary size | 3,528 |

Tomotopy LDA was trained on the train split only. Validation and test theta for
the LDA baseline were inferred with the trained tomotopy model. The neural
encoder was trained on the train split only, with fixed tomotopy beta, and the
best checkpoint was selected by validation loss.

## Test Split Score

Both models were exported with `--max-eval-motifs 0` and scored on the same
held-out test split.

| Model | Test spectra | Membership rows | Active motifs at 0.5 | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Tomotopy held-out inference | 200 | 115 | 84 | 0.083 | 0.5252 | 0.0436 |
| Fixed-beta neural encoder | 200 | 65 | 56 | 0.056 | 0.5397 | 0.0302 |

Additional encoder diagnostics:

| Metric | Value |
| --- | ---: |
| Validation theta cosine mean | 0.4948 |
| Validation top-1 agreement | 0.275 |
| Test theta cosine mean | 0.5086 |
| Test top-1 agreement | 0.275 |
| LDA test reconstruction NLL | 5.5032 |
| Neural test reconstruction NLL | 5.3814 |

## Interpretation

The neural encoder did not fully collapse: it produced 56 active held-out test
motifs at membership threshold `0.5`. However, it did not meet the smoke-run
success criterion. Mean SoS was slightly higher than held-out LDA, but coverage
and QAC were materially lower. Neural QAC was about 69% of the tomotopy
held-out baseline, outside the target 5-10% band.

The full MSn cache was not run because this 2k smoke result is not credible
enough to justify the full benchmark.

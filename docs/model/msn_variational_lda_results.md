# MSn Variational LDA Results

## 2026-05-20 2k no-tomotopy EM runs

These runs test a split-aware LDA-like EM model that learns:

```text
phi[d,v,k]   proportional to theta[d,k] * beta[k,v]
theta[d,k]  proportional to alpha + sum_v X[d,v] * phi[d,v,k]
beta[k,v]   proportional to eta   + sum_d X[d,v] * phi[d,v,k]
```

No tomotopy LDA beta, LDA theta, or neural encoder is used. `beta` is trained on
the train split only. Validation and test spectra get local theta inference with
`beta` frozen. Export uses the standard benchmark contract: `theta.npy`,
`beta.npy`, `vocab.json`, and `split_indices.json`.

Input cache:

```text
notebooks/Paper_results/MSn_evaluation/msn_cache_2000_fixed_beta
```

Runs:

```text
notebooks/Paper_results/MSn_evaluation/variational_lda_2000_k500_nmf
notebooks/Paper_results/MSn_evaluation/variational_lda_2000_k500_random
```

Configuration shared by both runs:

| Setting | Value |
| --- | ---: |
| Spectra | 2,000 |
| Split | 80/10/10 |
| Motifs | 500 |
| EM iterations | 100 |
| Held-out theta inference iterations | 50 |
| Alpha | 0.1 |
| Eta | 0.01 |
| Background weight | 0.05 |
| Export theta power | 3.0 |
| Seed | 42 |

## Test Split Score

Both models were exported with `--max-eval-motifs 0` and scored on the same
held-out test split.

| Model | Init | Best iteration | Test membership rows | Active motifs at 0.5 | Beta effective support | Coverage | Mean SoS | QAC |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Variational LDA, 500 motifs | NMF | 80 | 112 | 96 | 28.6 | 0.192 | 0.5685 | 0.1092 |
| Variational LDA, 500 motifs | Random | 1 | 154 | 131 | 1141.7 | 0.262 | 0.5487 | 0.1438 |

Reference context:

| Model | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: |
| LDA 2k | 0.592 | 0.5459 | 0.3232 |
| neural-lda 2k | 0.604 | 0.5344 | 0.3228 |
| Amortized neural, 500 motifs | 0.238 | 0.5194 | 0.1236 |

## Interpretation

The random-initialized EM run is the best no-LDA held-out result so far: QAC
`0.1438`, above the amortized neural 500-motif result of `0.1236`. However, it is
still far below the 2k LDA reference QAC around `0.323`, so it does not justify a
full-cache benchmark.

The random run's best checkpoint was iteration 1 and had very broad beta rows
with mean effective support around `1142`. Later EM iterations improved
reconstruction but reduced validation active motifs. This means the current
selection rule can find a better benchmark point than reconstruction-only
selection, but the model has not yet learned LDA-quality sparse motifs.

The NMF-initialized run learned much sharper beta rows with mean effective
support around `29`, but scored lower QAC (`0.1092`). This reinforces the current
diagnosis: beta shape and held-out motif coverage matter more than raw BoW
reconstruction, and sparse beta alone is not sufficient.

The next useful work is a targeted validation/checkpoint criterion and prior
sweep for this EM model, not another encoder architecture.

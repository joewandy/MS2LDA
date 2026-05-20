# MSn ProdLDA Results

## 2026-05-20 2k validation

This experiment tests a no-LDA ProdLDA/AVITM-style topic model for the MSn
benchmark. It is inspired by the Pyro ProdLDA tutorial but is implemented
directly in PyTorch to avoid adding a Pyro dependency.

The model learns:

```text
q(z | x) = Normal(mu_encoder(x), sigma_encoder(x))
theta    = softmax(z)
beta     = learned topic-word decoder logits
p(x)     = Multinomial(softmax(theta @ beta_logits))
```

Benchmark export uses deterministic encoder `theta = softmax(mu)`, a row-wise
softmax of `beta_logits` as `beta.npy`, the cache vocabulary, and the standard
`split_indices.json`.

Input cache:

```text
notebooks/Paper_results/MSn_evaluation/msn_cache_2000_fixed_beta
```

2k runs:

```text
notebooks/Paper_results/MSn_evaluation/prodlda_2000_k500_kl1
notebooks/Paper_results/MSn_evaluation/prodlda_2000_k500_kl01
```

Success threshold before considering larger runs:

- at least 100 active held-out test motifs at membership threshold `0.5`;
- held-out test QAC above the current no-LDA best of `0.1236`;
- ideally QAC around `0.20` or higher before scaling to the full MSn cache.

## Test split score

Both models were exported with `--max-eval-motifs 0` and scored on the same
held-out test split.

| Model | KL weight | Test spectra | Membership rows | Active motifs at 0.5 | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ProdLDA, 500 motifs | 1.0 | 200 | 72 | 1 | 0.000 | n/a | 0.0000 |
| ProdLDA, 500 motifs | 0.1 | 200 | 200 | 16 | 0.016 | 0.5481 | 0.0088 |

Reference context:

| Model | Coverage | Mean SoS | QAC |
| --- | ---: | ---: | ---: |
| LDA 2k | 0.592 | 0.5459 | 0.3232 |
| neural-lda 2k | 0.604 | 0.5344 | 0.3228 |
| Amortized neural, 500 motifs | 0.238 | 0.5194 | 0.1236 |

## Interpretation

The vanilla ProdLDA validation failed the benchmark criteria. The `kl_weight=1.0`
run collapsed to one active held-out motif. Reducing the KL weight to `0.1`
reduced the collapse but still produced only 16 active held-out motifs and a QAC
of `0.0088`, far below the current no-LDA amortized model.

This result does not rule out variational neural topic models for MSn, but this
direct ProdLDA form is not usable as a tomotopy replacement. The failure mode is
topic under-use and weak motif coverage, not bad substructure quality among the
few motifs that can be evaluated.

## Relationship to earlier work

The older neural replacement work used Poisson factorization / KL-NMF-like
objectives, sometimes with BoW or peak encoders and semi-amortized refinement.
The current `run_msn_amortized_neural_topic_experiment.py` also learns local
train-document `theta` parameters plus an encoder.

This ProdLDA run is different because there are no learned per-document topic
parameters. Topic proportions are latent logistic-normal variables inferred by
the encoder, and the training objective is an ELBO-like reconstruction plus KL
term. That makes it a cleaner test of whether an amortized neural variational
topic model can replace LDA-style inference without using LDA outputs.

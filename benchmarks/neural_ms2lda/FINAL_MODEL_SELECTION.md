# Final model selection: Contextual Sparse ETM

Status: **model frozen; clean held-out test reproduction complete**.

## Decision

The selected model is Contextual Sparse ETM. It is the published Embedded Topic
Model (ETM) with a channel-balanced decoder, one-scalar contextual top-2
posterior evidence and a published 1.5-entmax simplex mapping.

This is the simplest tested formulation that jointly preserves three required
properties on short mass-spectral documents:

- sparse per-spectrum topic mixtures;
- broad use of the global topic inventory; and
- a large chemically evaluable and useful Mass2Motif inventory.

The complete model has 19,278,001 learned parameters. Channel balancing and
entmax add no learned parameters; contextual evidence adds exactly one scalar
relative to balanced ETM.

## Held-out test results

Every neural model uses the same split, 21,233-word train-only vocabulary,
48-dimensional train-only SGNS coordinates, K=1,000, encoder width 800, raw
pseudo-count objective, Adam settings, 120 epochs and primary training seed
7043. Test matrices were released only after model and validation artifacts
were frozen.

| Model | Optimized | Evaluable | Useful | Mean SOS | Median SOS | NLL lower is better |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Canonical ETM | 601 | 171 | 101 | 0.631759 | 0.633333 | **8.686003** |
| Balanced ETM | **887** | 207 | 130 | 0.644232 | 0.640351 | 8.779686 |
| **Contextual Sparse ETM** | 799 | **572** | **343** | 0.637702 | 0.639500 | 9.535540 |
| Tomotopy LDA | 609 | 319 | 188 | **0.652752** | **0.651515** | 9.739090 |

Contextual Sparse ETM more than doubles the useful inventory relative to either
ETM control and produces 155 more useful motifs than Tomotopy. Its conditional
mean SOS lies between canonical and balanced ETM and below Tomotopy. Dense ETM
controls retain better completion NLL. The supported conclusion is therefore a
large discovery-breadth gain with explicit conditional-quality and predictive
fit trade-offs.

## Why each added component remains

Truth-known synthetic experiments separate the components. At K=36, contextual
evidence improves planted topic and document recovery and broadens topic use;
entmax then converts that evidence into exact sparse mixtures. Entmax without
context makes mixtures sparse by starving much of the fitted inventory.

The overcomplete K=128 seed-11 experiment is the strongest isolation check:

| Formulation | Planted motifs recovered at beta cosine >= 0.50 | Median exact support | Unique fitted top-1 topics |
| --- | ---: | ---: | ---: |
| Balanced ETM plus softmax | 6 / 18 | 128 | 7 |
| Balanced ETM plus 1.5-entmax | 2 / 18 | 2 | 3 |
| **Contextual Sparse ETM** | **18 / 18** | **2** | **19** |

The 18 recovered planted motifs and 19 fitted winner topics answer different
questions and must not be conflated. Recovery uses optimal truth matching;
winner count measures how many fitted topics are largest for at least one
spectrum.

Removing the learned context scalar and using direct token-only top-2 evidence
was also tested under the same synthetic protocol. It did not preserve recovery
or inventory breadth, so the one-scalar contextual formulation is retained.

## Initialization robustness

Only initialization and minibatch order change across the three final model
runs.

| Seed | Optimized | Evaluable | Useful | Mean SOS | NLL | Median effective topics | Unique top-1 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7043 | 799 | 572 | 343 | 0.637702 | 9.535540 | 3.680 | 917 |
| 23 | 798 | 582 | 353 | 0.631071 | 9.526086 | 3.714 | 922 |
| 37 | 805 | 557 | 327 | 0.628022 | 9.516054 | 3.744 | 914 |

Every seed is finite, records zero MAG clustering and optimization failures,
avoids a catastrophic duplicate component, and preserves sparse local mixtures
alongside broad global topic use. These are descriptive repeats on one fixed
split, not evidence of independent-dataset generalization.

## Executable model form

For normalized raw-count vector x_d, fixed word coordinates rho, learned ETM
topic coordinates alpha and K topics:

```text
beta       = channel_balanced_softmax(alpha @ rho.T)
r_d        = count_weighted_top2_contextual_evidence(x_d, rho, alpha, c)
offset_d   = center(log(r_d + 1/K))
mu_tilde_d = ETM_mu(x_d) + offset_d
z_d        = mu_tilde_d + exp(0.5 * logvar_d) * epsilon
theta_d    = entmax_1.5(z_d)
p(words|d) = theta_d @ beta
loss       = raw_count_multinomial_NLL + Gaussian_KL
```

Deterministic inference sets `z_d = mu_tilde_d`. The source functions and
report equations are mapped explicitly in Appendix B of the paper and checked
end to end by `tests/test_contextual_sparse_etm.py`.

## Evidence and integrity

The compact evidence package is
`research/etm_ecrtm_msnlib/local_results/20260901_contextual_sparse_etm_reproduction/`.
Its `acceptance.json` reports every predeclared scientific claim gate true;
`data_quality.json` reports pass; and `checkpoint_manifest.json` records the
reproduction ID, raw source commit and test-release boundary.

The canonical manuscript is `docs/research/neural_ms2lda_report.tex`, generated
from that package, with the reviewed PDF beside it.

## Final scope

The result supports Contextual Sparse ETM as a broad Mass2Motif discovery model
on one group-disjoint MSnLib split. It does not claim uniform metric dominance,
external-dataset generalization, expert-confirmed structural annotation or a
production-backend replacement. Those limits are explicit in the manuscript.

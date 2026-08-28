# Synthetic ETM + M1-component mechanism screen

Status: three-seed synthetic mechanism screen; not a real-data model-selection result.

## Question

Which M1 component should be added first to a published ETM-derived model before spending more real MSnLib compute?

The screen used the same short sparse paired-fragment/loss simulator as the faithful ETM campaign: 18 planted motifs, K=36 fitted topics, seeds 11/23/37, 800 training and 160 validation spectra per seed, train-only vocabulary, train-only 48D SGNS, and 120 ETM epochs.

Variants:

1. canonical fixed-SGNS ETM;
2. fragment/loss-balanced ETM;
3. canonical ETM + detached shared-geometry document gate;
4. balanced ETM + detached shared-geometry document gate;
5. balanced ETM + gate + prototype separation.

The gate reuses ETM's existing geometry and adds no learned parameter block. For normalized document BOW x, fixed SGNS rho and ETM topic vectors alpha, it computes `u=normalize(x @ rho)`, `g=softmax(2 u alpha^T / tau_g)`, and uses `theta_tilde ∝ theta_ETM * stopgrad(g)^gamma`. Gate temperature 1.0 and exponent 1.0 were selected on seed 11 from a bounded post-hoc screen and then frozen. Crucially, the useful result appears when this gate participates during training; post-hoc gating alone was only modest/mixed.

## Three-seed means

| Variant | NLL ↓ | true beta cosine ↑ | true theta cosine ↑ | active topics ↑ | nearest-beta redundancy ↓ | unique top-1 ↑ |
|---|---:|---:|---:|---:|---:|---:|
| canonical ETM | 6.4304 | 0.5224 | 0.8290 | 22.7 | 0.6739 | 15.0 |
| balanced ETM | 6.4217 | 0.5210 | 0.8262 | 23.7 | 0.6652 | 15.0 |
| canonical ETM + gate | 6.3779 | 0.5429 | 0.8314 | 31.7 | 0.5250 | 17.7 |
| balanced ETM + gate | 6.3770 | 0.5477 | 0.8302 | 32.3 | 0.5272 | 17.3 |
| balanced ETM + gate + separation(5) | 6.3483 | 0.5633 | 0.8275 | 32.0 | 0.4603 | 18.7 |

## Main result

The detached shared-geometry gate is the clear first addition. Relative to balanced ETM, balanced ETM + gate improves mean held-out NLL from 6.4217 to 6.3770, true-beta recovery from 0.5210 to 0.5477, active topics from 23.7/36 to 32.3/36, and nearest-topic redundancy from 0.6652 to 0.5272. Mean true-theta recovery is essentially preserved (0.8262 to 0.8302). The beta and redundancy improvements occur on all three seeds.

Canonical ETM + gate has almost identical aggregate performance, confirming that the gate itself is the key mechanism. However, balanced ETM + gate is more stable across seeds, and real MSnLib already showed that channel balancing raises optimized motif coverage from 609 to 911. Therefore the preferred real-data candidate is **fixed-SGNS ETM + 50/50 fragment/loss decoder + detached shared-geometry document gate**.

## Separation result

Prototype separation is not recommended in the first real run. On seed 11 it looked excellent, but the gains did not transfer cleanly to seeds 23 and 37. Weight 5 reduced redundancy further on average, yet worsened true-theta recovery on seeds 23/37 and worsened true-beta recovery on seed 37 relative to gate alone. A weaker weight 2 was also not robust. Separation should therefore remain a conditional second-stage intervention only if the real gated ETM still shows duplicate components.

## Decision

Run one real validation-only candidate first:

**Balanced fixed-SGNS ETM + detached shared-geometry document gate, tau_g=1.0, gamma=1.0.**

Do not add separation, NPMI, Sinkhorn, or token routing in that first run. If it still fails, diagnose the failed metric before adding one further mechanism.

This candidate remains recognizably ETM: the VAE document encoder and ETM embedding decoder are retained; the only MS-specific additions are the already-motivated channel normalization and a parameter-free gate that reuses the same SGNS/topic geometry.

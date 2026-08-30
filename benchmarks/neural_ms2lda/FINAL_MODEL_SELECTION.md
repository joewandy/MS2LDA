# Routing-informed ETM validation checkpoint

Status: **current paper-facing baseline, validation only**. Candidate test data
remain locked.

## Technical summary

Routing-informed sparse ETM is the first recognizable ETM-family candidate in
this study to combine sparse short-spectrum mixtures, a broad non-collapsed topic
inventory and competitive chemical discovery breadth. On the locked seed-42
MSnLib validation split it produces 445 evaluable and 289 useful motifs, exceeding
M1 by 37 and 24 and Tomotopy by 239 and 151, respectively.

The model does not dominate every metric. M1 retains more MAG-optimizable topics,
higher mean SOS and better completion NLL. Tomotopy has higher mean and median SOS
over a much smaller evaluable set. Routing ETM's median SOS is nevertheless higher
than M1's, and its completion NLL is better than Tomotopy's.

The predeclared all-gates result remains false because Routing ETM misses the
optimized, mean-SOS and NLL thresholds. Scientifically, this is a strong near-pass
and a viable baseline rather than an architecture failure. The old diffuse-theta
and global-collapse problems have been solved; remaining work concerns robustness
and a modest quality/likelihood trade-off.

Source implementation and result commit:
`3d9af674949a70a38cbd250b95023f28b9514fe5`.

## Routing ETM leads discovery breadth

All values below use the same seed-42 validation split, K=1000, membership
threshold 0.5 and leakage-controlled MAG/SOS protocol.

| metric | M1 | **Routing ETM** | Tomotopy |
|---|---:|---:|---:|
| optimized motifs | **884** | 803 | 607 |
| evaluable motifs | 408 | **445** | 206 |
| useful motifs | 265 | **289** | 138 |
| mean SOS | 0.658079 | 0.647153 | **0.676149** |
| median SOS | 0.648864 | 0.657895 | **0.685450** |
| completion NLL (lower is better) | **8.974140** | 9.542924 | 9.662228 |

"Better" therefore has a precise meaning: Routing ETM is better for discovery
breadth and better than Tomotopy for completion, but not uniformly better on
annotation coverage or SOS. This is the appropriate paper-facing claim.

The SOS distribution explains why Routing ETM can have a lower mean but a higher
median and more useful motifs than M1:

| validation SOS band | M1 | Routing ETM | difference |
|---|---:|---:|---:|
| high, SOS >0.8 | 79 | 67 | -12 |
| intermediate, 0.6–0.8 | 186 | 222 | +36 |
| low, SOS <0.6 | 143 | 156 | +13 |
| evaluable total | 408 | 445 | +37 |
| useful total, high + intermediate | 265 | 289 | +24 |

Routing ETM expands the intermediate-quality inventory substantially. Its
evaluable-to-optimized conversion is 55.4% versus M1's 46.2%, and its
useful-to-optimized conversion is 36.0% versus M1's 30.0%. The result is not an
artifact of reporting only a small, high-scoring subset.

## The model remains explainable ETM

The paper-facing generator is unchanged Embedded Topic Model:

- fixed train-only SGNS word embeddings;
- learned topic embeddings and the ETM embedding decoder;
- separate fragment/loss softmaxes with equal channel mass;
- multinomial reconstruction from the original intensity pseudo-counts;
- a Gaussian variational latent with analytic standard-normal KL; and
- the ordinary two-layer ETM document encoder.

The short-spectrum adaptation changes only posterior evidence and the final
simplex mapping. Each non-zero spectrum word receives a leave-one-out contextual
vector using one learned scalar. The vector is compared with ETM's own topic
coordinates, its strongest two topic matches are retained, and their weighted
document evidence is aggregated. For normalized spectrum `x`, ordinary ETM mean
`mu(x)`, aggregate route `r(x)` and K topics:

```text
mu_routed(x) = mu(x) + center(log(r(x) + 1/K))
z            = mu_routed(x) + sigma(x) * epsilon
theta        = entmax_1.5(z)
p(words | x) = theta @ beta_ETM
```

The `1/K` pseudocount bounds the log offset and row centring makes uniform
evidence an exact no-op. Alpha-entmax 1.5 supplies exact zeros. The only new
learned parameter is the context scalar; routing temperature and strength are
fixed at 1.0. Total parameters increase from 19,278,000 to 19,278,001.

There is no nonlinear M1 router, document gate, Sinkhorn balancing, NPMI loss,
prototype separation, alternating optimizer or temperature schedule. ETM and
entmax are published components. The top-2 contextual evidence is the single
domain adaptation justified by the observed failure of short spectra under
ordinary amortized ETM inference.

## Synthetic experiments isolate the complementary mechanisms

Truth-known synthetic spectra used 18 planted motifs, 1–3 generating motifs per
spectrum, paired fragment/loss words, train-only SGNS, raw intensity pseudo-counts
and frozen seeds 11, 23 and 37.

K=36 means:

| formulation | NLL | beta cosine | theta cosine | median effective | active >0.5% | unique top-1 |
|---|---:|---:|---:|---:|---:|---:|
| balanced ETM + softmax | 6.4710 | 0.3052 | 0.4868 | 3.59 | 8.0 | 6.7 |
| balanced ETM + entmax | 6.5977 | 0.2566 | 0.3973 | 1.88 | 5.0 | 4.7 |
| top-2 context + softmax | **6.2438** | 0.4423 | 0.7066 | 3.22 | 13.3 | 11.7 |
| **top-2 context + entmax** | 6.2892 | **0.4648** | **0.7513** | **2.00** | **14.0** | **13.0** |

Entmax alone makes theta sparse but worsens topic starvation. Routing alone
improves recovery and breadth but remains too diffuse under the separate
distinct-word stress. Their combination is complementary: routing supplies a
broad recoverable posterior signal and entmax converts it into sparse mixtures.

At seed-11 K=128, the selected combination recovered all 18 planted motifs as
distinct top-1 topics, obtained beta/theta recovery 0.6638/0.9556, median
effective topics 1.45, median exact support 2 and held-out NLL 6.1276. This
high-overcompleteness result supported the single real promotion.

## Real validation fixes both former ETM failures

| model | evaluable | useful | median effective topics | unique top-1 | principal failure |
|---|---:|---:|---:|---:|---|
| canonical fixed-SGNS ETM | 130 | 79 | 43.79 | 260 | diffuse, weak breadth |
| balanced ETM | 166 | 104 | 46.72 | 260 | diffuse, weak breadth |
| balanced gated ETM, gamma 2 | 220 | 138 | 65.57 | 471 | diffuse, weak breadth |
| entmax sparse ETM | 7 | 6 | 1.80 | 20 | sparse but starved |
| **Routing ETM** | **445** | **289** | **3.70** | **828** | residual quality/NLL trade-off |

The selected model's median exact support is 6, mean effective topics 4.09 and
p95 exact support 13. It has 538.40 corpus-effective topics, 535 topics above
0.0005 mean usage and maximum mean topic use only 0.0143. One two-topic component
appears at beta cosine 0.999; there is no catastrophic duplicate component.

This is the decisive change from earlier candidates: sparse per-spectrum use no
longer comes from collapsing the global inventory.

## The formal gate misses are retained as caveats

The gates were frozen before the real candidate was scored. Counts retain 95% of
M1, mean SOS retains 99%, and NLL may be at most 105% of M1. Every gate had to be
true for the Boolean pass.

| gate | threshold | Routing ETM | distance from gate | formal result |
|---|---:|---:|---:|:---:|
| optimized motifs | ≥840 | 803 | -37, or 4.40% below | fail |
| evaluable motifs | ≥388 | 445 | +57 | pass |
| useful motifs | ≥252 | 289 | +37 | pass |
| mean SOS | ≥0.651498 | 0.647153 | -0.004345, or 0.67% below | fail |
| completion NLL | ≤9.422847 | 9.542924 | +0.120077, or 1.27% above | fail |
| finite/stable | required | yes | — | pass |
| no catastrophic duplicate component | required | yes | — | pass |

The all-gates value must remain false for audit integrity. It should be reported
as a conservative selection outcome, not collapsed into the misleading statement
that Routing ETM is generally worse than M1.

## Scope, definitions and methodology

- **Optimized motif:** a topic for which MAG produced at least one optimized
  annotation feature; count equals annotation coverage times 1,000 topics.
- **Evaluable motif:** an optimized topic associated with at least one unique
  validation molecule through spectrum membership `theta >= 0.5`.
- **SOS:** annotation-fingerprint containment in the molecule fingerprint,
  averaged per unique molecule and then per evaluable topic.
- **Useful motif:** an evaluable topic with SOS at least 0.6.
- **Completion NLL:** negative log-likelihood per in-vocabulary held-out token
  under `theta @ beta`; lower is better.
- **Evidence population:** the frozen seed-42 MSnLib training/validation split,
  train-only vocabulary and SGNS, leakage-filtered MAG index and 3,889 validation
  spectra. Candidate test was not accessed.

Training used deterministic CUDA, 120 fixed epochs, Adam 0.005, weight decay
1.2e-6, batch 256 and six CPU threads. Training took 876.8 seconds on the RTX
5070. Peak PyTorch allocated/reserved CUDA memory was 0.821/1.053 GB; process
high-water memory was 2.879 GB. Deterministic full-validation inference processed
23,426 spectra/second.

## Historical alternatives remain useful negative evidence

- Canonical ETM had good completion but too few chemically evaluable topics and
  very diffuse document mixtures.
- Fragment/loss balancing raised optimized coverage but did not repair evaluable
  or useful breadth.
- Post-hoc temperature sharpening changed mixture scale but could not jointly
  preserve breadth, SOS and likelihood.
- Entmax without routing produced exact sparsity by starving most topics.
- Pooled projected models contained a 614-topic near-exact duplicate component.
- The maintained ECRTM Sinkhorn path failed its convergence contract at real
  K=1000 and V=21,233; no partial model was scored.
- The reference short-document NSTM implementation failed the paired high-K
  synthetic screen and was not promoted to MSnLib.

These experiments justify the selected mechanism without claiming that every
future topic model must fail.

## Robustness, limitations and checkpoint integrity

The central limitation is real-seed uncertainty: Routing ETM has one completed
real training seed on one data split. Synthetic mechanism behavior is confirmed
over seeds 11, 23 and 37, but real multiseed stability is not yet established.
No test-set performance claim is made.

The committed checkpoint contains source, configs, synthetic summaries, real
metrics, per-topic chemical scores, training history, diagnostics, environment,
validation-access audit and provenance. Large weights, arrays, MAG indexes and
raw data stay outside Git with paths, sizes and SHA-256 hashes recorded.

Run the machine check without training:

```bash
conda run -n ms2lda-neural python \
  scripts/verify_routing_etm_checkpoint.py
```

On the original host, add `--verify-inputs --verify-local-artifacts
--require-external` to hash every retained input and large artifact. The full
check completes 78 consistency/integrity checks. Exact replay commands are in
`research/etm_ecrtm_msnlib/local_results/20260830_routing_etm/README.md`.

## Recommended next steps

1. Preserve this checkpoint as the baseline for every future comparison.
2. Repeat the unchanged Routing ETM on two additional real training seeds using
   validation only; do not tune against their outcomes.
3. If the optimized-coverage and mean-SOS gap is stable, consider one bounded
   train-derived positive-NPMI experiment. Stop if it reduces the present
   evaluable/useful breadth, sparse support or global inventory.
4. Keep the private M1 architecture and its other mechanisms out of the
   paper-facing model.
5. Unlock test only after an independent review freezes the method, checkpoint
   and success interpretation.

## Further questions

- Are 445 evaluable and 289 useful motifs stable across real training seeds?
- Which topics account for the 197 unoptimized Routing ETM components, and are
  they unused noise, chemically coherent novel motifs or decoder artefacts?
- Does the lower mean SOS persist after controlling for the candidate's 37-topic
  increase in evaluable breadth?
- Can one coherence intervention recover optimized coverage without sacrificing
  NLL or recreating M1's complexity?
- After stability review, does the frozen candidate reproduce its advantage on
  the untouched test split?

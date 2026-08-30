# ETM / ECRTM direction for Neural MS2LDA

## Current next direction

The current paper-facing candidate is a published ETM base with top-2 contextual
posterior evidence and alpha-entmax 1.5. It is not M1 and does not use the M1
gate, Sinkhorn, separation, alternating optimizer or schedule.

Its completed K=1000 validation has sparse, broad topic use (median 3.70
effective topics; 828 unique top-1 topics) and passes the evaluable/useful gates
at 445/289. It narrowly fails optimized coverage, mean SOS and completion NLL,
so candidate test remains locked. The next bounded experiment adds only
train-derived positive-NPMI coherence regularization to this ETM formulation.
M1 multiseed is not the next campaign. See `NEXT_AGENT.md` and
`local_results/20260830_routing_etm/README.md`.

Status: **routing-informed ETM validation complete; bounded coherence follow-up**.

Branch: `codex/unified-ms2lda-environment`

## Why this branch exists

The current M1 neural model works well enough to be scientifically interesting, but it is a bespoke combination of token-level contextual routing, top-2 assignments, whole-spectrum evidence, a detached document gate, Sinkhorn balancing, positive-NPMI regularisation, prototype separation, temperature annealing and alternating optimisation.

That combination is defensible as an engineering result, but it is harder to motivate in a computational-biology paper than a recognizable published neural topic model with one or two clearly motivated mass-spectrometry adaptations.

The objective of this branch is therefore not to simplify M1 mechanically. It is to answer a cleaner question:

> Can MS2LDA move from classical LDA to an established neural topic-model family, and add only the minimum MS-specific change that real MSnLib data demonstrate is necessary?

The preferred published lineage is:

1. **ETM** — Dieng, Ruiz & Blei, *Topic Modeling in Embedding Spaces*, TACL 2020.
2. **ECRTM** — Wu, Dong, Nguyen & Luu, *Effective Neural Topic Modeling with Embedding Clustering Regularization*, ICML 2023.

## Canonical code references

### ETM

Reference repository: `adjidieng/ETM`.

The baseline used here was mechanically based on Adji Dieng's original July 2019 code, commit:

`cbb67bf484282e66df00cd2166bf8dc740a95a1d`

The relevant published ETM ingredients are retained:

- a two-layer document encoder;
- Gaussian variational posterior over document logits;
- logistic-normal topic proportions `theta = softmax(z)`;
- standard-normal KL;
- topic embeddings and word embeddings defining topic-word probabilities through inner products;
- ordinary bag-of-words reconstruction;
- either jointly learned or fixed pretrained word embeddings.

For MS2LDA, **fixed train-only SGNS embeddings** are the preferred baseline because that is a recognized ETM mode and it naturally reuses the repository's spectral co-occurrence embeddings.

### ECRTM

Reference repositories:

- standalone ICML implementation: `bobxwu/ECRTM`;
- maintained implementation: `bobxwu/TopMost`.

The maintained TopMost implementation keeps the published model equations and is the preferred implementation reference. At the time of this study it uses:

- two-layer softplus VAE encoder;
- logistic-normal approximation to a symmetric Dirichlet prior;
- trainable word embeddings initialized from pretrained embeddings;
- trainable topic embeddings;
- negative Euclidean topic-word distance;
- `beta_temp = 0.2`;
- Embedding Clustering Regularization (ECR) implemented through balanced Sinkhorn optimal transport;
- maintained default `weight_loss_ECR = 100` and `sinkhorn_alpha = 20`.

## Evidence boundary

### Already established on real MSnLib

The committed M1 result remains the incumbent real-data evidence.

Validation, seed 42, K=1000:

| metric | M1 |
|---|---:|
| optimized motifs | 884 |
| high-confidence evaluable motifs | 408 |
| useful motifs | 265 |
| mean SOS | 0.6580793714 |
| median SOS | 0.6488636364 |
| validation completion NLL | 8.9741399256 |

The repository's direct M1 ablations also show that, **within the M1 architecture**, removing individual mechanisms can be harmful. In particular the committed simplification ledger reports large losses from removing the document gate, Sinkhorn, positive-NPMI and prototype separation. Those results remain valid; this branch changes model family rather than claiming those mechanisms were unnecessary in M1.

### Established only on synthetic MS/MS-like data

ETM and ECRTM have been tested on truth-known simulated spectra designed specifically around the short sparse MS2LDA representation. These results are architecture-screening evidence, **not chemical validation**.

### Not yet established

Neither fixed-SGNS ETM nor ECRTM has yet been run through the full real MSnLib validation MAG/SOS pipeline in this branch. Do not claim otherwise.

## Synthetic MS/MS regime

The simulator is deliberately unlike a typical NLP corpus:

- 18 planted Mass2Motifs;
- 800 train / 160 validation / 160 test spectra per seed;
- seeds 11, 23 and 37;
- 18-42 physical peaks per spectrum, median about 29;
- median only about 25-26 non-zero in-vocabulary fragment/loss terms;
- fragment and neutral-loss words generated as paired observations from the same physical peak;
- completion splitting by whole physical peak, preventing fragment/loss leakage;
- only 1-3 generating motifs per spectrum;
- long-tailed motif prevalence;
- shared/ambiguous motif anchors;
- common background peaks and noise;
- train-only vocabulary and train-only SGNS embeddings;
- intensity converted to the MS2LDA pseudo-count `round(100 I)`.

A crucial characteristic is that a spectrum can have only about 25 non-zero terms while carrying roughly 1,400 total pseudo-count mass. This matters for VAE objectives because reconstruction is scaled by those pseudo-counts while KL is not.

## Finding 1: vanilla learned-embedding ETM can collapse

The clearest stress test deliberately over-specifies the model: K=36 fitted to data generated from 18 true motifs.

Across seeds 11, 23 and 37, **jointly learned-embedding ETM** shows strong component/topic collapse:

- only about **8/36** topics receive >0.5% mean corpus usage;
- corpus effective topic count about **9.67**;
- nearest-topic redundancy cosine about **0.851**;
- true-theta recovery about **0.506**;
- true-beta recovery about **0.403**;
- nevertheless held-out NLL is good, about **6.153**.

This is an important failure mode: predictive likelihood can look good while much of the nominal topic inventory is unused or duplicated.

This is better described as **component/topic collapse** than classic posterior collapse: the variational latent representation is still used, but learned topic components become redundant and/or starved.

## Finding 2: fixed spectral SGNS is already a strong ETM inductive bias

At K=36, original ETM with fixed train-only SGNS embeddings is much more robust:

- 27/36 materially active topics;
- true-beta recovery: **0.521**;
- true-theta recovery: **0.834**;
- median effective topics per spectrum: **5.91**;
- nearest-topic redundancy: **0.664**;
- held-out NLL: **6.441**.

At correctly specified K=18:

- all 18 topics are materially active;
- true-beta recovery: **0.439**;
- true-theta recovery: **0.784**;
- median effective topics per spectrum: **4.88**;
- redundancy: **0.480**;
- held-out NLL: **6.532**.

This makes fixed-SGNS ETM the strongest simple published baseline to take to MSnLib first.

## Finding 3: published ECRTM fixes topic inventory collapse

TopMost-style ECRTM was initialized with the same train-only SGNS vectors. No M1 routing, document gate, fragment/loss balancing, NPMI or custom prototype-separation penalty was added.

Three-seed means:

| K | model | NLL | true beta cosine | true theta cosine | median effective topics/spectrum | materially active topics | nearest-topic redundancy |
|---:|---|---:|---:|---:|---:|---:|---:|
| 18 | fixed-SGNS ETM | 6.532 | 0.439 | **0.784** | 4.88 | 18/18 | 0.480 |
| 18 | learned-embedding ETM | **6.112** | 0.377 | 0.586 | 2.06 | 11.7/18 | 0.676 |
| 18 | TopMost-style ECRTM | 6.756 | **0.457** | 0.549 | 15.34 | 18/18 | **0.477** |
| 36 | fixed-SGNS ETM | 6.441 | 0.521 | **0.834** | 5.91 | 27/36 | 0.664 |
| 36 | learned-embedding ETM | **6.153** | 0.403 | 0.506 | 2.28 | 8/36 | 0.851 |
| 36 | TopMost-style ECRTM | 6.759 | **0.550** | 0.647 | 29.95 | **36/36** | **0.511** |

ECRTM therefore does what it was designed to do in the stress condition:

- keeps the full topic inventory occupied;
- reduces topic duplication;
- improves recovery of the planted motif-word distributions.

But it exposes a different mismatch with MS/MS: its document mixtures are much too diffuse for spectra generated from only 1-3 motifs.

## Finding 4: changing ECRTM Dirichlet alpha alone does not fix diffuse theta

Changing ECRTM's symmetric Dirichlet concentration from alpha=1.0 to alpha=0.1 had little effect in a representative synthetic run.

The likely reason is objective scale. With MS2LDA pseudo-count intensities, reconstruction is roughly 9,000-13,000 per short spectrum, while KL is only about 25-30. The reconstruction term therefore dominates strongly despite the document having only a few dozen non-zero terms.

This is a domain-representation issue worth measuring on real MSnLib.

## Finding 5: a predeclared inference-only theta temperature is effective in simulation

For ECRTM, define

`theta_tau = softmax(log(theta) / tau)`.

No model weights or beta are changed.

Tau=0.30 was selected on synthetic seed 11 and then frozen unchanged for seeds 23 and 37.

K=36 three-seed means:

| theta tau | NLL | true theta cosine | median effective topics/spectrum |
|---:|---:|---:|---:|
| 1.00 | 6.759 | 0.647 | 29.95 |
| 0.50 | **6.444** | 0.774 | 14.95 |
| 0.40 | 6.504 | 0.794 | 9.34 |
| **0.30** | 6.699 | **0.805** | **4.74** |
| 0.25 | 6.866 | 0.807 | 3.29 |
| 0.10 | 7.779 | 0.792 | 1.36 |

Tau=0.30 is therefore a reasonable **predeclared synthetic-derived calibration candidate** if real ECRTM theta is again unusably diffuse. It must not be tuned on the MSnLib test split.

## Finding 6: do not create a new ETM+ECR hybrid

A direct transplant of the ECR penalty onto fixed-SGNS ETM performed poorly in a representative K=36 seed-11 experiment:

- only 6 materially active topics;
- nearest-topic redundancy about 0.922.

It is both less defensible scientifically and empirically worse than using the published ECRTM model. Do not pursue this as the main paper path.

## Finding 7: ECRTM scaling is a real concern at MSnLib K and V

The synthetic experiments use roughly K<=36 and a vocabulary around two thousand words. Real MSnLib is approximately:

- K = 1000;
- V = 21,233.

ECRTM's ECR term operates on a full K x V cost/kernel matrix and the released training loop evaluates ECR repeatedly during minibatch optimisation. At real scale this is a qualitatively larger computation than the synthetic experiment.

The standalone ECRTM solver also showed seed-dependent runtime pathology with its exact 1,000-iteration Sinkhorn cap even at synthetic scale. A 50-step bounded approximation was therefore used for the multi-seed screen.

A representative synthetic exact-vs-bounded comparison was very close:

- exact cap 1000: NLL 6.7114, beta recovery 0.5480, theta recovery 0.6291, redundancy 0.5483;
- cap 50: NLL 6.7157, beta recovery 0.5439, theta recovery 0.6283, redundancy 0.5493.

This supports using the bounded solver as a **research numerical approximation**, but it must be labelled honestly. A higher-compute agent should first measure full-scale feasibility instead of silently assuming canonical ECRTM is practical at K=1000.

## Recommended real-data candidate ladder

For a computational-biology paper, keep the story recognizable:

1. **Fixed-SGNS ETM** as the main simple published neural baseline.
2. **TopMost ECRTM initialized from the same SGNS embeddings** as the published anti-collapse model.
3. Evaluate **ECRTM raw theta** and the already-predeclared `tau=0.30` theta calibration if raw mixtures are too diffuse.
4. Only if these recognizable models fail the real chemical gates should a new MS-specific mechanism be introduced.

M1 remains the incumbent/control throughout.

## Real-data validation gates

Use validation only for candidate selection. The test split remains locked until a candidate passes.

Provisional chemistry-first gates relative to the committed M1 validation result:

| metric | M1 | provisional candidate gate |
|---|---:|---:|
| optimized motifs | 884 | >= 840 (95%) |
| evaluable motifs | 408 | >= 388 (95%) |
| useful motifs | 265 | >= 252 (95%) |
| mean SOS | 0.658079 | >= 0.651498 (99%) |
| completion NLL | 8.974140 | <= 9.422847 (105%) |
| finite/stable | yes | required |

These gates are a practical handoff rule for this candidate comparison. If the repository's existing locked protocol contains a stricter directly applicable rule, the existing rule takes precedence.

## What is already on this branch

- `scripts/run_published_topic_models_msnlib.py`
  - research-only ETM/ECRTM real-data runner;
  - reuses the exact repository split/vocabulary/SGNS preparation;
  - currently performs validation-only training, completion evaluation and collapse diagnostics;
  - intentionally does not open the test split.
- `.github/workflows/etm-ecrtm-msnlib-research.yml`
  - **manual workflow_dispatch only**;
  - no automatic execution on pushes;
  - useful as a reproducible feasibility runner, not yet the preferred full chemical campaign environment.
- `research/etm_ecrtm_msnlib/repro/`
  - synthetic simulator and faithful ETM/ECRTM screening scripts.
- `research/etm_ecrtm_msnlib/results/`
  - exact synthetic CSV evidence used for the findings above.

See `NEXT_AGENT.md` for the real-data execution instructions and scientific stopping rules.

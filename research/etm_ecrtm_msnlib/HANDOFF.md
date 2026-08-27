# Neural MS2LDA published-model handoff

This is the canonical handoff for the next higher-compute research/implementation agent. It complements `NEXT_AGENT.md`; where they overlap, the scientific stopping rules below and the locked repository evaluation protocol take precedence.

## Read the literature survey before changing the model

Before implementing or proposing another architecture, read:

- `research/etm_ecrtm_msnlib/LITERATURE_SURVEY.md`
- `research/etm_ecrtm_msnlib/REFERENCES.bib`
- `research/etm_ecrtm_msnlib/README.md`
- `research/etm_ecrtm_msnlib/NEXT_AGENT.md`

The survey is part of the model-selection evidence, not background decoration. It maps the MS2LDA problem onto the relevant published families: classical MS2LDA and MS2LDA 2.0, MSnLib, Spec2Vec, ETM and scETM, AVITM/ProdLDA component collapse, short-document topic models, Neural Sinkhorn Topic Model, ECRTM, sparse-Dirichlet VAEs, NPMI/co-occurrence regularisation, Contextualized Topic Models, DreaMS and FASTopic. It also maps every model variant already explored in this research campaign onto that literature.

The reviewer-facing default should therefore be **a published model plus the smallest domain adaptation justified by a specific real-data failure**. Do not add another bespoke mechanism merely because it improves one simulation metric.

## Problem statement

MS2LDA documents are unusually short and sparse compared with ordinary NLP documents. A typical spectrum has only a few dozen physical peaks and activates only a small number of latent substructure motifs, yet intensity quantisation can produce a large pseudo-count mass. Fragment and neutral-loss words are chemically coupled observations of the same physical peaks. The desired output is not only low held-out likelihood: the model must retain a broad, non-redundant inventory of chemically useful Mass2Motifs and assign spectra strongly enough for the existing MAG/SOS workflow.

This combination creates two distinct failure modes that must not be conflated:

1. **topic/component collapse** — nominal topics become unused or learn near-duplicate word distributions even when likelihood is good;
2. **over-diffuse document mixtures** — the topic inventory is healthy, but each short spectrum spreads probability over far too many topics.

Our simulations reproduced both failure modes under controlled truth.

## Why the current M1 model is not discarded

The committed M1 model remains the incumbent because it has real MSnLib chemical evidence. At seed 42 and K=1000 its validation result is 884 optimized motifs, 408 high-confidence evaluable motifs, 265 useful motifs, mean SOS 0.6580793714, median SOS 0.6488636364 and completion NLL 8.9741399256.

M1's direct real-data ablations also show that several mechanisms matter **inside that architecture**. Removing the document gate, Sinkhorn, positive-NPMI or prototype separation caused substantial motif losses. This handoff does not reinterpret those ablations as mistakes. The research question is instead whether a different, established topic-model family can obtain comparable chemistry with a simpler and more recognisable computation graph.

## Evidence established in simulation

The simulator was deliberately designed around tokenised MS/MS rather than generic text: 18 planted motifs; 800/160/160 train/validation/test spectra per seed; seeds 11, 23 and 37; 18–42 physical peaks with median around 29; median only about 25–26 non-zero in-vocabulary fragment/loss words; paired fragment/loss observations; whole-peak document-completion splitting; 1–3 active generating motifs; long-tailed motif prevalence; ambiguous/shared anchors; background/noise peaks; train-only vocabulary and SGNS; and intensity pseudo-counts `round(100 I)`.

### ETM

A faithful ETM baseline based on Adji Dieng's original implementation was tested in both recognised modes: fixed pretrained embeddings and jointly learned embeddings. In the overcomplete K=36 experiment fitted to 18 true motifs, learned-embedding ETM showed strong topic/component collapse: about 8/36 materially used topics and nearest-topic beta cosine around 0.851 despite good held-out NLL around 6.153.

Fixed train-only spectral SGNS was itself a strong inductive bias. At K=36 it retained about 27/36 materially active topics, true-beta recovery about 0.521, true-theta recovery about 0.834, median effective topics per spectrum about 5.91, redundancy about 0.664 and held-out NLL about 6.441. At K=18 all 18 topics remained active.

This makes **fixed-SGNS ETM the first real-data candidate**, not jointly learned-embedding ETM.

### ECRTM

TopMost-style ECRTM was initialised from the same train-only SGNS embeddings, without importing M1 routing, its document gate, NPMI or its custom prototype-separation loss. In the K=36 stress experiment it kept 36/36 topics occupied, reduced redundancy to about 0.511 and improved true motif-word recovery to about 0.550. Thus ECRTM solves the topic-inventory failure it was designed for.

Its failure on our synthetic spectral representation was different: raw ECRTM used about 30 effective topics per spectrum, far too diffuse for spectra generated from only 1–3 motifs. Lowering the symmetric Dirichlet concentration from 1.0 to 0.1 barely changed this. The likely cause is objective scaling: roughly 1,400 pseudo-count mass per short spectrum makes reconstruction much larger than the KL term.

An inference-only calibration `theta_tau = softmax(log(theta) / tau)` was therefore tested. Tau 0.30 was selected on seed 11 and then frozen unchanged for seeds 23 and 37. Across the three K=36 seeds it reduced median effective topics from about 29.95 to 4.74 and raised true-theta recovery from about 0.647 to 0.805, while leaving beta and the ECR anti-collapse geometry unchanged. This is a predeclared candidate calibration, not permission to tune tau on the real test split.

### Negative controls

A direct transplant of ECR onto fixed-SGNS ETM was both less defensible and empirically poor: a representative K=36 run left only 6 materially active topics with redundancy around 0.922. Do not pursue this hybrid as the main path.

The standalone ECRTM Sinkhorn solver also showed runtime sensitivity. A representative exact-1000-iteration versus bounded-50-step comparison was scientifically very close in the synthetic run, but the bounded version is a numerical research approximation and must be labelled as such. At real K=1000 and V≈21,233, ECR's full topic-word transport matrix is a major computational concern that should be measured explicitly.

## Literature-guided candidate ladder

The next experiment should remain deliberately narrow:

1. **Fixed-SGNS ETM** — the cleanest established neural extension of LDA/ETM for the current representation. scETM strengthens the computational-biology precedent for embedded topic models on sparse biological counts.
2. **TopMost ECRTM with the same SGNS initialisation** — the published anti-collapse candidate if ETM loses topic inventory or produces excessive duplicates.
3. **Raw ECRTM and the already-frozen tau=0.30 theta calibration** — only to address the distinct short-spectrum mixture-calibration problem; do not tune tau on test.
4. Add one further MS-specific mechanism only if the real validation result identifies a specific failure. For example, if beta predicts completion reasonably but MAG/SOS coherence falls, positive-NPMI is a targeted literature-supported mechanism to test. If representation quality is the bottleneck, a frozen DreaMS/contextual embedding is a separate later experiment, not something to combine immediately with ECRTM.

M1 remains the incumbent/control throughout.

## Real-data experiment: validation only

Use the repository's existing deterministic MSnLib split, train-only vocabulary, train-only SGNS feature construction, document-completion protocol and leakage-filtered MAG/SOS evaluation. Candidate selection is **validation only**. Do not inspect candidate test results until a candidate passes the predeclared validation rule.

The provisional chemistry-first gates relative to M1 are:

| metric | M1 validation | candidate gate |
|---|---:|---:|
| optimized motifs | 884 | >= 840 |
| evaluable motifs | 408 | >= 388 |
| useful motifs | 265 | >= 252 |
| mean SOS | 0.658079 | >= 0.651498 |
| completion NLL | 8.974140 | <= 9.422847 |
| finite/stable | yes | required |

If an existing locked repository rule is stricter and directly applicable, the existing rule wins.

For every candidate, also report collapse and sparsity diagnostics: corpus topic usage, effective topic count per spectrum, maximum/nearest beta similarity, active-topic counts and fitting/inference time. Good NLL alone is not evidence that the model is scientifically usable.

## Code and result map

Current production-facing/reference code on this branch:

- `scripts/run_published_topic_models_msnlib.py` — validation-only real-data research runner using repository data preparation.
- `research/etm_ecrtm_msnlib/repro/published_models_reference.py` — compact reference implementations of ETM/ECR/ECRTM equations used in this study.
- `research/etm_ecrtm_msnlib/repro/SIMULATION_PROTOCOL.md` — simulator and metric specification.
- `research/etm_ecrtm_msnlib/results/` — machine-readable synthetic summaries and negative controls.

Historical work predating the final published-model direction is retained as provenance under `research/etm_ecrtm_msnlib/archive/`. It includes the initial M1 simplification campaign, the faithful ETM collapse campaign, the ECRTM follow-up, candidate code, ablation scripts, tests, report source and result tables. The current literature-guided ladder above supersedes those earlier provisional recommendations, but their code/results must remain available for audit.

## Scientific stopping rule

Do not search indefinitely. If fixed-SGNS ETM passes the chemical gates, it is preferred for paper simplicity unless ECRTM provides a clear, reproducible chemical advantage that justifies its cost. If ETM fails because of topic starvation/duplication, evaluate ECRTM. If ECRTM retains topics but has diffuse spectrum mixtures, evaluate only the frozen theta calibration. If all recognisable published candidates fail a chemistry gate, diagnose the failure and add **one** targeted mechanism supported by the literature and the failed metric.

The goal is not to make the highest-scoring neural architecture. The goal is the simplest scientifically defensible model that preserves Mass2Motif discovery quality on real MSnLib data.

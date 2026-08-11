# Neural-assignment MS2LDA: bounded research protocol

## Research question

Can a genuinely fully neural model discover a large set of reusable
Mass2Motifs without variational Bayes, conjugate topic updates, DreaMS inputs,
or iterative inference for a new spectrum?

This is a second-generation attempt, not a continuation of the failed
document-level neural VAE. The earlier model could reconstruct held-out
spectra but used too few coherent topics. Its negative result and the broader
technical explanation remain in
[fully_neural_ms2lda_challenges.md](fully_neural_ms2lda_challenges.md).

## What changes

The earlier model compressed a whole spectrum into one dense latent topic
vector. The new model instead makes the topic decision at the level where the
evidence occurs:

1. each unique fragment or neutral-loss token is projected into a learned
   128-dimensional space;
2. the token is combined with the same spectrum after that token's count mass
   has been removed;
3. the resulting routing vector is compared with every learned topic
   prototype;
4. a sparse top-two assignment is made in one pass;
5. the spectrum's topic mixture is the count-weighted sum of those token
   assignments.

The topic-word distribution is not a separate free matrix. It is derived from
the same topic prototypes and projected token table. This couples the
explanation of a token to the topic that emits it.

In compact notation, for token \(w\) in spectrum \(d\),

\[
r_{dw} =
\operatorname{normalize}\left(
e_w + f_\psi\left[e_w,\,
\frac{\sum_{v\ne w} c_{dv}e_v}{\sum_{v\ne w}c_{dv}}\right]
\right),
\]

\[
q_{dw} = \operatorname{TopKSoftmax}
\left(\frac{r_{dw}^{\mathsf T}p_k}{\tau}\right),\qquad
\theta_d =
\frac{\sum_w c_{dw}q_{dw}}{\sum_w c_{dw}},
\]

\[
\beta_{kw} =
\operatorname{softmax}_w
\left(\frac{2\,p_k^{\mathsf T}e_w}{\tau_\beta}\right).
\]

Inference is exactly this routing and aggregation. There is no local
optimization loop.

## Inputs and exclusions

The token table combines the same train-only 48-dimensional SGNS embedding
used in the preserved v1 experiment with Fourier m/z features and a
fragment/loss indicator, giving 64 input dimensions. Topic prototypes use
deterministic weighted k-means++ seeding without Lloyd iterations. A token's
weight is the square root of its corpus frequency multiplied by squared
inverse document frequency. This prevents ubiquitous background tokens from
consuming the initial prototype budget while remaining entirely train-only.

The candidate receives no chemical structure, scaffold, compound identity,
MAG result, DreaMS vector, Tomotopy topic, NMF factor, or classical-topic
teacher. Tomotopy is read only as an external comparator after fitting.

## Training signal

Four deterministic pairs of training views each retain 80% of physical peak
groups. A fragment and its neutral-loss token are always retained or removed
together. Retained intensities are renormalized without using the omitted
maximum.

Each epoch alternates two blocks.

- Router blocks minimize symmetric cross-view completion NLL, a balanced
  Sinkhorn assignment cross-entropy, and cross-view topic-mixture consistency.
  The exact topic-word distribution is cached and detached during the block.
- Four topic blocks minimize exact cross-view completion NLL plus a local
  token-to-topic decoder loss with weight 0.25. Routing assignments are
  detached during these blocks.

Training uses a straight-through sparse top-two forward assignment. Evaluation
uses deterministic top two directly. Sinkhorn targets are stop-gradient
targets; the Sinkhorn operator itself is differentiable and tested for its
balanced marginals.

## Bounded anti-collapse safeguards

The primary routing temperature decreases from 0.5 to 0.1 over 30 epochs.
Sinkhorn weight is 0.25 for ten epochs, decreases to 0.05 by epoch 40, and
then remains fixed. Cross-view consistency weight is 0.1.

A topic used below 0.1 times uniform for three validation points may be
replaced by a highest-loss routing context through epoch 60. Replacement is
deterministic, is limited to twice per topic, and clears the matching Adam
state rows.

Only one rescue is eligible, and only for a validation-diagnosed collapse.
It restarts from the identical initialization. Underuse increases the
Sinkhorn and recycling strengths. Diffuse mixtures use top one and a final
temperature of 0.07. If both diagnoses occur, both predefined changes apply.
There is no likelihood-only rescue and no third attempt.

## Staged gates

The runner progresses automatically only after a stage passes.

1. Two K=32 synthetic problems require stability, at least 90% nontrivially
   occupied topics, matched beta cosine at least 0.75, top-20 Jaccard at least
   0.50, and completion NLL at least 10% better than unigram.
2. K=200 on MSnLib validation requires at least 120 active topics at the
   \(1/K\) threshold, diversity at least 0.65, NPMI at least -0.60, median
   effective topics from 2.3 to 36.5, and NLL at most 10.73.
3. K=1000 validation permits one primary and at most one collapse-only rescue.
   The selected attempt is frozen before test matrices are opened.
4. MAG/SOS is run only if the selected model passes every non-chemical test
   gate.

A genuine synthetic or K=200 failure stops expensive progression. If the
eligible K=1000 attempts fail, the result is preserved and the project returns
to fully neural model-design discussion. The runner does not automatically
redirect the research to downstream motif annotation.

## Relation to established ML work

The components are grounded in established methods:

- classical admixture semantics come from
  [latent Dirichlet allocation](https://www.jmlr.org/papers/v3/blei03a.html);
- competition among reusable prototypes is related to
  [Slot Attention](https://proceedings.neurips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)
  and vector-quantized representation learning;
- balanced stop-gradient assignment targets follow the principle used by
  [SwAV](https://proceedings.neurips.cc/paper/2020/hash/70feb62b69f16e0238f741fab228fec2-Abstract.html);
- neural-topic collapse motivates explicit topic geometry and balanced use,
  as in
  [ECRTM](https://proceedings.mlr.press/v202/wu23c.html).

The exact combination of physical MS/MS peak-group views, leave-one-token-out
spectral context, count-weighted sparse routing, and shared neural beta
geometry is our applied research synthesis. The experiment tests that
hypothesis; the ingredients alone do not establish novelty or success.

## Operational contract

Training uses four CPU threads and evaluation one. Batch sizes 64, 128, and
256 are benchmarked with complete router forward/backward blocks; the fastest
option below 8 GB RSS is frozen. The runner uses atomic epoch checkpoints,
durable heartbeats, launch-token verification, immutable input and code
manifests, and exact restart from the last completed epoch. It refuses to
start the scientific study before this implementation is merged into a clean
fork main.

# Why fully neural MS2LDA is difficult

## Short answer

A neural MS2LDA can retain the same scientific idea as classical LDA:

\[
p(w\mid d)=\sum_k \theta_{dk}\beta_{kw}.
\]

Each spectrum is still an admixture of Mass2Motifs, and each motif is still a
distribution over fragments and neutral losses. The hard part is not the
definition of the model. It is learning a large collection of distinct,
reusable motifs using ordinary gradient optimization.

Classical LDA repeatedly assigns observed tokens to topics and converts those
assignments into expected counts. Rare but consistent fragment patterns
therefore keep receiving a direct learning signal. A neural encoder and
decoder instead receive a reconstruction gradient. Early in training, a few
broad topics can explain many common peaks. Those topics then receive more
gradient, while unused topics receive little signal and die or become copies.

This document records why that matters for MS/MS, what our experiments have
already ruled out, and the bounded model we will test next.

## Two different collapses

**Posterior collapse** occurs when the spectrum encoder stops conveying useful
document-specific information. Its output approaches the prior or a nearly
constant distribution.

**Topic or component collapse** occurs when the global topic matrix contains
far fewer useful motifs than requested, or many topics become near-duplicates.
A model can avoid posterior collapse while still suffering topic collapse.

Reconstruction likelihood, NLL, or NPMI alone cannot establish that collapse
has been avoided. A collapsed model can reconstruct frequent fragments or
obtain apparently coherent common-peak groups. Topic activity, diversity,
duplication, per-spectrum sparsity, held-out prediction, and chemical utility
must be inspected together.

## Why MS/MS makes the problem harder

MS2LDA asks for roughly 1,000 reusable components from short, sparse spectra.
The corpus has several properties that amplify neural shortcuts:

- Most spectra contain only a small fraction of the 21,233-token vocabulary.
- Common fragments and neutral losses dominate reconstruction gradients.
- Rare motifs may be chemically valuable despite contributing few tokens.
- Motif prevalence is highly imbalanced; uniform topic use is not a valid
  target either.
- Useful spectra should contain sparse mixtures, while the corpus should use a
  diverse global set of motifs.
- Chemically equivalent evidence can vary with precursor, instrument, missing
  peaks, and noise.

At K=1000, simply adding a generic VAE or a larger MLP gives the optimizer more
ways to reconstruct spectra without guaranteeing 1,000 distinct chemical
building blocks.

## What our previous experiments showed

The preserved amortized-LDA experiments at commit
`b12278a72ed18594bfede0b6c82f6cab212e48f9` provide strong negative controls.

- Random-start amortized LDA used 1 of 200 topics. Its token NLL was 7.919,
  versus 4.952 for the Tomotopy reference, and top-20 diversity was 0.061.
- A beta-only NMF initialization retained 71 of 200 topics but remained much
  worse than the reference.
- Fully warm-starting beta and the encoder retained 200 topics, but joint
  stochastic training did not improve the non-neural initialization.
- Earlier no-LDA neural experiments reached held-out QAC 0.1438, versus 0.3232
  for the 2,000-iteration LDA reference.

The anchored semi-amortized model prevented activity collapse, but it did so
by retaining classical local coordinate updates and the conjugate global
expected-count update. Across its five mushroom runs, encoder-only inference
was fast and all 200 topics were used, but matched cross-run topic cosine was
0.414 versus 0.566 for Tomotopy. It therefore demonstrated useful neural
inference, not a stable fully neural replacement for topic discovery.

These results rule out repeating ProdLDA, adding capacity to the same
random-start amortized model, or assuming that longer training will repair the
problem.

## What the simplification study established

The completed v8 study evaluated two discovery modes, five inference modes,
four local-refinement budgets, two held-out splits, and two spectrum
representations. Its verifier retained 226 required artifacts.

The most defensible simplification was DreaMS-informed discovery followed by a
topic-only direct encoder and one local VB correction. Relative to the original
DreaMS-dependent encoder with two VB steps, it:

- improved validation and test completion NLL by 0.78% and 0.82%;
- retained 351 and 352 active topics, versus 354 for the reference;
- reduced learned inference parameters and checkpoint size by 22.3%; and
- reduced warm model-only latency by 31.7% and 36.2%.

It is not fully neural: its topics were already frozen, its direct targets came
from 50-step VB, and inference retained one VB correction. The zero-VB arms
tested only inference against frozen topics, not whether neural topic discovery
would collapse.

DreaMS-informed and symmetric-prior discovery produced nearly identical topic
matrices in this run (mean matched cosine 0.9992 and mean top-word Jaccard
0.9637). Removing DreaMS entirely nevertheless worsened completion NLL by
about 3.1–3.3%. This is useful simplification evidence, but it does not provide
a compelling fully neural discovery result.

## The bounded next attempt

The next model will preserve the admixture likelihood while making both topic
discovery and spectrum inference gradient-based. It will use:

- training-only fragment/loss co-occurrence embeddings augmented by m/z and
  token type;
- a sparse spectrum encoder producing topic proportions in one pass;
- a neural topic-token matrix with exact multinomial reconstruction; and
- embedding-clustering optimal transport to prevent topics from occupying the
  same semantic region.

The anti-collapse mechanism follows the motivation of
[ECRTM](https://arxiv.org/abs/2306.04217), adapted to sparse MS/MS and an
LDA-like mixture decoder. It will be implemented independently from the paper,
without copying its reference repository.

There will be one primary configuration and one predeclared rescue that adds
bounded corpus-usage and document-sparsity guards. The rescue is eligible only
after genuine collapse; it is not a route to open-ended tuning.

## Decision boundary

The hard question is whether the method is viable, not whether its first run
immediately dominates every mature reference. A working fully neural model
must remain numerically stable, retain a substantial non-duplicated topic set,
produce non-uniform but sparse spectrum mixtures, stay within a relaxed
held-out likelihood bound, and retain measurable chemical utility.

Stricter likelihood, coherence, coverage, chemical-quality, and speed targets
will be reported as a competitive scorecard. Missing one of those targets will
not erase a viable neural result. If neither the primary nor the eligible
rescue passes the viability gates, neural work will move from motif discovery
to Mass2Motif annotation, cross-dataset matching, and substructure retrieval.

## Relevant primary references

- Srivastava and Sutton, [Autoencoding Variational Inference for Topic
  Models](https://arxiv.org/abs/1703.01488).
- Dieng, Ruiz, and Blei, [Topic Modeling in Embedding
  Spaces](https://aclanthology.org/2020.tacl-1.29/).
- Wu et al., [Effective Neural Topic Modeling with Embedding Clustering
  Regularization](https://arxiv.org/abs/2306.04217).
- Hoyle et al., [Is Automated Topic Model Evaluation Broken? The Incoherence
  of Coherence](https://aclanthology.org/2021.neurips-main.143/).
- Doan and Hoang, [Are Neural Topic Models
  Broken?](https://aclanthology.org/2022.findings-emnlp.390/).

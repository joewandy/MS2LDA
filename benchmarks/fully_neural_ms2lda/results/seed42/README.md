# Fully neural MS2LDA result checkpoint

## Decision

Neither the primary model nor the predeclared collapse-only rescue is a viable
MS2LDA topic-discovery model. Stop this bounded discovery direction and redirect
neural work to Mass2Motif annotation, cross-dataset motif matching, and
substructure retrieval using proven Tomotopy topics.

This is not a failure of neural inference speed or held-out reconstruction. It
is a failure to discover and use the requested large collection of coherent,
reusable motifs.

## Result

| Metric | Tomotopy | Primary | Rescue | Hard requirement |
| --- | ---: | ---: | ---: | ---: |
| Test completion NLL/token | 9.7569 | 8.6251 | 8.6480 | no worse than 10.7326 |
| Corpus-active topics | 363 | 45 | 54 | at least 254.1 |
| Top-10 word diversity | 0.8148 | 0.7413 | 0.7331 | at least 0.6648 |
| Mean training-corpus NPMI | -0.2997 | -0.7027 | -0.6979 | competitive target -0.3497 |
| Median full-spectrum effective topics | 9.1173 | 30.9945 | 62.3776 | 2.2793 to 36.4692 |
| Cached one-pass speedup | 1x | 10,761x | 9,911x | competitive target 2x |

The primary improved NLL by 11.60% relative to Tomotopy, and the rescue improved
it by 11.37%. Those apparently strong results did not translate into motif
discovery: the models used only 12.4% and 14.9% as many corpus-active topics as
the reference. Their NPMI was also roughly 0.40 lower.

The primary passed the relaxed mixture-width gate on full spectra but still used
only 45 topics globally. The rescue increased activity to 54 topics, while
making individual mixtures substantially more diffuse and failing the
mixture-width gate. The predeclared usage and sparsity guards therefore changed
the collapse, but did not solve it.

MAG/SOS chemistry was not run. Both candidates had already failed the
non-chemical hard gates, so chemical evaluation could not rescue the decision
and would have spent compute without changing viability.

## Interpretation

The experiment separates three claims that would otherwise be easy to confuse:

1. A neural network can predict held-out spectral tokens very well.
2. A one-pass neural encoder can be extremely fast.
3. The model can discover roughly 1,000 distinct, coherent, reusable chemical
   motifs.

The first two claims succeeded. The third did not. Reconstruction rewards a
small collection of broad components that explain common fragments. ECR kept
the topic-word rows reasonably diverse, but it did not make the encoder use
most of them, and the rescue guards traded some extra usage for overly diffuse
document mixtures. This is the collapse mechanism described in
`docs/research/fully_neural_ms2lda_challenges.md`, now observed at K=1000 on the
fixed MSnLib split.

## Run and provenance

- Tracking issue: [#6](https://github.com/joewandy/MS2LDA/issues/6)
- Reviewed implementation PR: [#7](https://github.com/joewandy/MS2LDA/pull/7)
- Redirect issue: [#8](https://github.com/joewandy/MS2LDA/issues/8)
- Frozen fork-main revision: `de3a497b04682809d96cd7a795f6c743a6bd28bd`
- Full local run: `/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/fully-neural-ms2lda-seed42-v1`
- Full run size: 405 MB across 64 files
- Training-only SGNS: 5.42 seconds
- Primary: 124 epochs, 935.98 seconds, early stopping
- Rescue: 132 epochs, 1,072.50 seconds, early stopping
- CPU policy: four training threads and one evaluation thread

Both live-source and frozen-source provenance verification passed. Seventeen
generated initialization, model, and evaluation artifacts matched their
recorded SHA-256 values. An idempotent resume rehearsal returned the identical
decision without retraining.

This compact directory retains the protocol, run lock, code manifest, candidate
audit, training histories, selected training summaries, evaluation summaries,
gate tables, and final report. Large beta, theta, checkpoint, and model files
remain in the full local run; their hashes are recorded in the retained
completion manifests.

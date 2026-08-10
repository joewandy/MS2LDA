# Fully neural MS2LDA bounded experiment

This package implements the experiment frozen in
[issue #6](https://github.com/joewandy/MS2LDA/issues/6). It tests one fully
neural topic-discovery model and, only if pre-test validation diagnostics show
topic collapse, one collapse-guarded rescue from the exact same initialization.

The candidate uses training-only SGNS token features, m/z Fourier/type features,
a one-pass logistic-normal encoder, a neural topic/token decoder, and sampled
ECR-style optimal-transport regularization. It has no DreaMS dependency, local
VB, conjugate update, Tomotopy/NMF warm start, or iterative held-out inference.

The exact multinomial likelihood is optimized in alternating blocks for CPU
tractability: encoder minibatches use a cached exact normalized topic matrix,
then the topic/token geometry receives an exact full-vocabulary normalization
and decoder update. No sampled-softmax approximation is used.

Use the durable wrapper rather than invoking the package directly:

```bash
scripts/run_fully_neural_ms2lda.sh start
scripts/run_fully_neural_ms2lda.sh status
scripts/run_fully_neural_ms2lda.sh resume
scripts/run_fully_neural_ms2lda.sh verify
```

Training uses four CPU threads. Evaluation uses one thread to match the retained
Tomotopy latency protocol. Every epoch is checkpointed, logs are appended, and
re-running `resume` verifies the frozen code and inputs before continuing.

Hard viability decides whether a working neural discovery model exists. The
stricter competitive scorecard is descriptive: it records where a viable model
still trails Tomotopy without automatically killing the research direction.

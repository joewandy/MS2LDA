# Neural-assignment MS2LDA

This isolated research benchmark is the second bounded attempt at a genuinely
fully neural MS2LDA discovery model. It preserves the earlier document-level
neural experiment and the current hybrid model unchanged.

The candidate routes each observed fragment or loss token to learned topic
prototypes in one pass, then forms spectrum topic mixtures by count-weighted
aggregation. Topic-word distributions come from the same prototype and token
geometry. Training alternates router blocks and four exact neural topic blocks.

The candidate does not use DreaMS, variational Bayes, conjugate updates,
Tomotopy or NMF initialization, classical topic teachers, chemistry metadata,
or iterative held-out inference. Differentiable Sinkhorn targets,
stop-gradient targets, and deterministic dead-prototype recycling are the only
non-gradient safeguards.

The unattended study is dependency-gated:

1. two K=32 synthetic recovery problems;
2. a K=200 MSnLib validation screen;
3. a K=1000 primary attempt and at most one collapse-only rescue;
4. one test evaluation of the validation-selected attempt;
5. chemical evaluation only after non-chemical viability passes.

A genuine synthetic or K=200 failure stops progression. If both eligible
K=1000 attempts fail, the artifacts are preserved and the project returns to
fully neural model-design discussion. There is no automatic redirect to motif
annotation.

Use scripts/run_neural_assignment_ms2lda.sh for start, resume, status, verify,
and smoke operations. The long scientific run refuses to start unless this
implementation has been merged into a clean fork main.

The architecture, literature grounding, exact gates, and stop rule are
documented in docs/research/neural_assignment_ms2lda_protocol.md.

The completed v1 run stopped only at the K=200 active-topic screen. Its
post-hoc exploratory K=1000 continuation is a separate committed protocol:
`protocol_k1000_continuation.json`. It records the raw K=200 failure, waives
only that screening failure as blocking, and leaves every final K=1000, test,
and chemical criterion unchanged. The rationale and interpretation boundary
are documented in
`docs/research/neural_assignment_ms2lda_k1000_continuation.md`.

Use `scripts/run_neural_assignment_ms2lda_k1000_continuation.sh` for the
continuation's `start`, `resume`, `status`, and `verify` operations. It fixes
the amended protocol, run directory, and detached-session identity together.

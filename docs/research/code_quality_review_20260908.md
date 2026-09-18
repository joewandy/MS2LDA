# Scientific code-quality review — 8 September 2026

This record describes the first completed code-quality pass, before the later
sharing revision. The follow-up [equation audit](equation_code_audit_20260908.md)
and [current report source inventory](contextual_sparse_etm_report_sources.md)
document the subsequent equation checks and repeated baseline comparison.
The earlier verification counts and numerical caveats below are preserved as
history, not presented as the latest report state.

## Outcome and scope

The thermo-nuclear maintainability review and an independent scientific-code
review were completed before consolidating the work into one upstream-based
commit. The user's scientific-code preference governed the refactor: ordinary
functions, explicit arrays/tensors and explanatory mathematics, not additional
object-oriented layers. The selected whole-spectrum model and published ETM
MLP backbone are retained. Tomotopy remains the production backend.

The review covers the branch's research models, training/evaluation runners,
data and evidence boundaries, report generators, tests, packaging and existing
production changes relative to upstream. The final research lint scope contains
75 Python files. The inherited production corpus correction uses the loss
channel's own intensities instead of accidentally pairing losses with fragment
intensities; its regression coverage remains in `tests/test_generate_corpus.py`.
The maintained production suite also exercises the current modeling pipeline.

## Main findings and fixes

| Finding | Resolution |
| --- | --- |
| A 1,057-line packager mixed evidence verification, model-specific extraction, aggregation and publication | Reduced the CLI to 178 lines. Comparison and summary functions have separate canonical modules; repeated chemistry columns use one function. No framework or service classes were added. |
| Training/evaluation scripts imported other CLIs as libraries and duplicated loaders, inference, model construction and JSON/hash operations | Moved reusable operations into the research package. Runners now call those functions. Package modules do not import `scripts`; an AST regression enforces the direction. |
| A validation manifest flag could be accepted without rechecking the consumed input bytes or complete manifest coverage | Recheck each required source/link before opening count matrices; enforce validation vocabulary width and record alignment. Reject invalid counts and role-swapped input identities. |
| Test release could leave partially exposed inputs after a later failure | Preflight all sources/destinations, then roll back only links created by the failed call. Evaluation rechecks complete test input seals and frozen model ownership. |
| A synthetic inventory check counted runs globally, allowing duplicated per-model seeds or an unexpected topic count | Check the exact multiset of `(K, seed, formulation)` records before aggregation. Existing historical criteria are not reinterpreted as model-selection rules. |
| Chemical scoring could silently omit or duplicate topic annotations | Require one annotation per topic, aligned spectrum records and a valid fingerprint threshold. Valid-data scores are unchanged. |
| Provenance and platform assumptions were too permissive | Include scientific data/metric dependencies in source identities, check runtime/source compatibility before exact resume, fail closed if Git identity cannot be established, and normalize platform-dependent memory units. |
| A dictionary cache still recomputed each repeated molecule's connectivity key | Compute it only on cache misses; annotation values are unchanged. |
| Model documentation obscured current versus archived equations | Added tensor dimensions, equation labels, normalization axes, posterior/KL conventions and the selected whole-spectrum path to comments/docstrings and the package README. |

No research Python file exceeds 1,000 lines. The remaining small PyTorch model
classes own learned state and preserve historical checkpoint names and seeded
initialization. Reordering initialization or deleting archived switches would
break paired experiments without simplifying the selected mathematical model.

## Mathematical correction: uniform-evidence derivative

One finding is a real numerical correction, not behavior-preserving cleanup.
For `K` topics, the contextual offset is

```text
delta_k(r) = log(r_k + 1/K) - mean_j log(r_j + 1/K).
```

At uniform evidence, `r_k = 1/K`, its value is zero but its Jacobian is
`(K/2) (I - 11^T/K)`. The previous exact-zero `torch.where` branch returned a
constant tensor there, incorrectly making its derivative zero. The corrected
implementation subtracts only a detached forward-roundoff correction, keeping
the exact-zero value and the analytic derivative. Regression tests probe
simplex-tangent directions for `K = 2, 4, 7`.

**Saved training results and checkpoints were not retrained or replaced.**
The forward map is unchanged, including its uniform-evidence value. Future
optimization may differ if it reaches an exactly uniform evidence row. The
review does not establish whether that boundary was encountered in historical
training; historical runs must be replayed with their recorded source snapshots
when an exact trajectory is required. This qualification applies to the shared
offset used by both the archived and selected contextual models.

## Equation-to-code reading guide

The [research package README](../../benchmarks/neural_ms2lda/README.md) maps the
current manuscript's operations to functions. The most important conventions
are documented directly in the implementation:

- normalized pseudo-counts enter the encoder, while **raw** pseudo-counts enter
  the multinomial reconstruction loss;
- decoder normalization is across words, with half-mass in each channel;
- whole-spectrum context includes the scored word; top two is per word;
- contextual evidence shifts the Gaussian mean and therefore changes both
  reconstruction and Gaussian KL, without adding an auxiliary penalty;
- `logvar` means log variance; the KL is for the latent Gaussian, not an
  asserted Gaussian density over the entmax simplex;
- deterministic inference uses entmax of the mean, not an expectation of
  entmax under the posterior.

## Verification record

All commands ran in the native Linux `ms2lda-neural` environment unless noted.

| Check | Observed result |
| --- | --- |
| Scientific regression suite | 232 passed, including CPU and CUDA checkpoint/recovery checks; no skips |
| Maintained production suite, `NUMBA_DISABLE_JIT=1` | 124 passed; two existing Lark deprecation warnings |
| Workflow-declared Black and research-config Ruff | All 75 research Python files pass |
| Independent before/after numerical fixture | 34 formulations: identical initialized state, sampled mixtures, decoder probabilities, losses and parameter-gradient hashes on the same fixed nonuniform fixture |
| Wheel build, payload inspection and isolated installed import | Pass; production CLI present, research package/runners absent from the wheel |
| MkDocs build in an isolated documentation environment | Pass; existing production docstring/link warnings remain, not silently suppressed |
| Four report generators, standard library only (`python -S`) | Pass; two successive generations are byte-identical for all 24 generated artifacts |
| Evidence versus the pre-review backup | No raw result or frozen evidence changed; only generator-source hashes in three review summaries changed |
| LaTeX and PDF | 14 pages; no build warnings, undefined references or overflowing boxes. Renamed first page visually inspected; pages 2–14 pixel-identical to the previously inspected report |

The independent numerical comparison used 29 ETM/reduction variants and five
published-model implementations with identical input tensors, initialization
and sampling noise in separate pre-/post-review processes. Both result files
have SHA-256
`9c7b34209eae3e555f64087561674da13e22dad7100c95eaf3be9fafd62b66d2`.
This is bounded fixture parity, not proof of all-input equivalence. The
uniform-evidence derivative is deliberately corrected and tested separately.

The updated report PDF has SHA-256
`a38004e5f8876d3b271f921ce64536326b805c341c1322db7e797d9c7c7767a3`.
The archived 29-page PDF remains unchanged. No full benchmark retraining,
new test-set evaluation or external publication was performed during this
code-quality pass. The scientific limitations in the manuscript remain:
development-selection reuse, single-fit baselines and proxy chemical metrics.

## Recovery and history

The pre-review commit is preserved locally as
`backup/pre-thermonuclear-20260908`. The original dirty source/evidence snapshot
is retained under `tmp/thermonuclear-20260908-Tvie3s/source-before.tar.gz`, with
SHA-256 `ae59ae18ab0ec0ee7efdfbbc56f80a365352f9b4f723df83807c514b5e84b3c6`.
This local recovery archive is intentionally excluded from Git.

The experimental branch is consolidated by amending its single commit, whose
parent is upstream revision `645a081`. The reviewed fork `main` checkpoint is
left intact. No force-push or upstream push is part of this operation.

## Sharing-revision follow-up

The later thermo-nuclear pass retained the same function-first approach. Its
bounded changes strengthen experimental bookkeeping and undefined-input
handling, without adding model components or a new class hierarchy:

- Synthetic matching now rejects invalid shapes, nonfinite/nonpositive mass
  and insufficient fitted topics. All six saved selected-model matching
  dictionaries are exactly unchanged on their valid inputs.
- Baseline caches identify the fit seed, recipe and consumed input bytes, and
  reject missing required artifacts. Tomotopy additionally checks checkpoint
  and evaluation-output hashes instead of silently refitting a damaged cache.
  Test evaluation still requires the canonical release gate, which is tested
  on a small fixture without opening study test data.
- A baseline stage records its attempt before launching. Launch/bookkeeping
  failures leave an explicit failed record and reap only that stage's child.
- Evidence publication is a small plain-function transaction: verify first,
  stage task-owned copies, seal them, then rename the complete directory.
  Failure-injection tests preserve original runs and prevent partial final
  bundles or replacement of existing evidence.
- The standard-library report summary requires the exact six-fit baseline
  inventory and shared evaluation denominators. It computes fit means and
  sample SDs, highlights only directional metrics, and checks the portable
  source manifest before generating any report input.

The [equation audit](equation_code_audit_20260908.md) maps every mathematical
operation, tests the complete selected forward calculation and derivatives,
and distinguishes numerical safeguards from the probabilistic model. In
particular, it proves the earlier uniform-evidence gradient branch unreachable
for the chosen real and synthetic `K=128` training inputs; the historical
`K=36` trajectory caveat remains explicit. No selected-model fit was replaced.

These defensive changes were made after the new baseline jobs' immutable
execution snapshot. The evidence records that training source separately from
the later postprocessing source; it does not attribute the final checkout's
subsequent edits to already-running fits. The current source inventory records
the final sharing-revision test, rendering and evidence checks.

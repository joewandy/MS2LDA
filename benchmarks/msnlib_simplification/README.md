# HybridLDA simplification study

This benchmark collects a frozen seed-42, K=1000 factorial comparison without
changing the supported `ms2lda_hybrid` API and without selecting or adopting a
replacement model.

## Frozen matrix

The two discovery modes are the corrected bounded DreaMS prior and a classical
symmetric `eta + expected counts` prior. Each is paired with:

- DreaMS plus topic evidence, trained with the two-step semi-amortized ELBO;
- DreaMS plus topic evidence, trained by direct posterior regression;
- topic evidence only, trained with the semi-amortized ELBO;
- topic evidence only, trained by direct posterior regression; and
- an analytic topic-evidence initializer with no learned encoder.

Every arm is evaluated at 0, 1, 2, and 50 local-VB steps. Common references are
selected per document by the highest 50-step local ELBO across every trained
initializer, the analytic initializer, and a uniform initializer. Validation is
fully frozen before the one posthoc test pass begins.

The symmetric discovery worker and count-only inference workers never construct
a DreaMS extractor or read the DreaMS feature cache. The three symmetric-prior
count-only arms are therefore fully DreaMS-free.

## Overnight operation

From this worktree, run:

```bash
scripts/run_hybrid_simplification_overnight.sh start
scripts/run_hybrid_simplification_overnight.sh status
```

`start` freezes source, code, configuration, and input hashes; verifies at least
20 GiB free disk and AC power; then launches a detached `screen` session under
`caffeinate -i`. Keep the Mac plugged in with its lid open. The same command with
`resume` restarts an interrupted run from verified checkpoints.

Preparation, neural-encoder training, and symmetric topic discovery use four
CPU threads. Final inference, scoring, and latency measurements use one thread
so evaluation remains directly comparable with the corrected source run.

The runner writes:

- `run_state.json`, `events.jsonl`, and a five-minute `heartbeat.json`;
- one durable log per task plus `overnight.log`;
- two retained checkpoint generations for long discovery and encoder stages;
- atomic completion manifests for every discovery, arm, split, and report; and
- `overnight_complete.json`, which is successful only when every required
  artifact verifies.

Independent arms continue after a worker failure. Dependent tasks are recorded
as skipped, and the runner exits nonzero. A genuine non-convergence is retained
as a scientific failure and is never automatically retuned.

## Result bundle

The neutral result bundle contains validation and test theta arrays, document
completion NLL, activity, diversity, NPMI, common-reference agreement, OOV data,
timings, parameter/checkpoint sizes, full-spectrum SOS inputs and results,
same-seed topic matching, and 2,000 fixed-seed scaffold-group bootstrap
replicates. Per-document and per-compound files retain the denominators needed
for later analysis.

No collection command ranks arms, declares a winner, changes a default, or
authorizes replacement of Tomotopy or the current Hybrid model.

## Completed checkpoint

The compact v8 result checkpoint is stored in `results/seed42-v8`. The exact
source used by that run is preserved at commit `fab2b78` on branch
`archive/hybrid-simplification-seed42-v8`. Use `verify-archive` for a completed
historical bundle after the live development source has changed; ordinary
`verify` remains strict so an in-progress run can never resume under drifted
code.

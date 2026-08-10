# HybridLDA simplification v8 checkpoint

This directory is the compact, reviewable checkpoint for the completed
seed-42, K=1000 simplification study. It contains the executed notebook,
self-contained HTML report, derived tables, neutral result tables, and the
integrity and recovery records needed to audit the conclusions.

The exact source frozen at the start of the run is preserved on branch
`archive/hybrid-simplification-seed42-v8`, commit `fab2b78`. All 53 entries in
`code_manifest.json` match that archive. The main development version contains
a later equivalent fingerprint-cache optimization; it is intentionally not
misrepresented as the code that produced the run.

The full result directory is 4.8 GiB and is not stored in Git. It remains at:

```text
/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/hybrid-lda-simplification-seed42-v8
```

`verification.json` commits to its 226 required artifacts through
`inventory_sha256`. The compact checkpoint retains every report-level input
and output, while excluding large topic matrices, document posteriors, model
checkpoints, and repeated chemical-association rows.

The run's chemical-scoring recovery is recorded in
`recovery_provenance.json`. It reused completed inference and annotations,
changed no model, and used the frozen environment plus the two retained
recovery scripts.

Verify the full local bundle against its frozen source with:

```bash
HYBRID_SIMPLIFICATION_RUN_DIR=/Users/joewandy/Work/data/MS2LDA-msnlib-validation/runs/hybrid-lda-simplification-seed42-v8 \
  scripts/run_hybrid_simplification_overnight.sh verify-archive
```

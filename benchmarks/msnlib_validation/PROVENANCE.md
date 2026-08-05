# Historical-code provenance

No archive branch was merged or cherry-picked. Two small, provenance-safe ideas
were selectively reimplemented against the current APIs:

- `archive/msn-evaluation:scripts/msn_benchmark_pipeline.py` supplied the
  precedent for deterministic Tomotopy construction and
  `model.train(..., workers=1, parallel=1)`. The confirmatory profile retains
  that setting. The separately labelled indicative profile deliberately uses
  the published notebook's `workers=0, parallel=3` training setting and records
  that the resulting Tomotopy fit is not bitwise reproducible. The current
  driver rewrites the orchestration, split discipline, metrics, manifests, and
  checkpointing.
- `archive/private-msn-eval:scripts/evaluate_motif_substructure_quality.py`
  supplied the historical SOS arithmetic: MACCS bits shared by the annotation
  and molecule divided by annotation bits. The current implementation adds
  shape/empty guards and tests.

No private repository metadata, generated result, absolute archive path, model,
or data artifact was copied. Both preserved branches remain unmodified.

The resumable Hybrid training format, two-generation fallback, four-thread
training/one-thread evaluation boundary, derived-protocol audit, and
hash-audited reuse support are new code written for this validation. They were
not copied from either archive. The current corrected run starts from an empty
run directory and reuses none of the superseded result artifacts. The raw-DreaMS
worker remains isolated from the legacy environment's duplicate OpenMP runtime;
no unsafe duplicate-runtime override is used.

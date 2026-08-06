# Historical-code provenance

No archive branch was merged or cherry-picked. Two small, provenance-safe ideas
were selectively reimplemented against the current APIs:

- `archive/msn-evaluation:scripts/msn_benchmark_pipeline.py` supplied the
  precedent for deterministic Tomotopy construction and
  `model.train(..., workers=1, parallel=1)`. The historical confirmatory
  profile retains that setting. The separately labelled indicative profile
  deliberately uses the published notebook's `workers=0, parallel=3` training
  setting and records that the resulting Tomotopy fit is not bitwise
  reproducible. The current driver rewrites the orchestration, split
  discipline, metrics, manifests, and checkpointing.
- `archive/private-msn-eval:scripts/evaluate_motif_substructure_quality.py`
  supplied the historical SOS arithmetic: MACCS bits shared by the annotation
  and molecule divided by annotation bits. The current implementation adds
  shape/empty guards and tests.

The paper repository's deposited `Benchmark_MAG_MSn.ipynb` and
`Analysis_MSnLib.ipynb` were inspected as primary methodology sources. They
establish that the published analysis used full fitted documents, a 0.5
membership heuristic, and MAG-optimized motifs. No notebook code was copied.
The current driver adapts those steps to held-out full spectra and additionally
reports a rank-based dominant-topic diagnostic without an absolute cutoff. It
uses the benchmark's frozen MACCS/0.8 setting equally for both methods, whereas
the downstream paper analysis notebook recomputed RDKit fingerprints at 0.9.
Accordingly, paper SOS values are contextual, not directly comparable. The
driver reports both the notebook's annotation-containment denominator and the
supplement's smaller-fingerprint denominator because those sources disagree.

No private repository metadata, generated result, absolute archive path, model,
or data artifact was copied. Both preserved branches remain unmodified.

The resumable Hybrid training format, two-generation fallback, four-thread
training/one-thread evaluation boundary, derived-protocol audit, and
hash-audited reuse support are new code written for this validation. They were
not copied from either archive. The corrected chemical child run starts from an
empty run directory and hash-reuses only the completed corrected-alpha feature
cache and core models; it does not reuse the removed fixed-alpha artifacts or
the superseded half-spectrum chemical outputs. The raw-DreaMS worker remains
isolated from the legacy environment's duplicate OpenMP runtime; no unsafe
duplicate-runtime override is used.

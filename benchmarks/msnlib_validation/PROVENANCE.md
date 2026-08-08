# Historical-code provenance

No archive branch was merged or cherry-picked. Two small, provenance-safe ideas
were selectively reimplemented against the current APIs:

- `archive/msn-evaluation:scripts/msn_benchmark_pipeline.py` supplied the
  precedent for deterministic Tomotopy construction and
  `model.train(..., workers=1, parallel=1)`. The historical confirmatory
  profile retains that setting. The published notebook used
  `workers=0, parallel=3`; the separately labelled indicative laptop profile
  preserves partition parallelism but caps the requested workers at six and
  records that the resulting Tomotopy fit is not bitwise reproducible. The
  current driver rewrites the orchestration, split
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

The resumable Hybrid and Tomotopy training formats, atomic feature-pool
generations, four-thread training/one-thread evaluation boundary,
derived-protocol audit, and hash-audited reuse support are new code written for
this validation. They were not copied from either archive. Independent review
found that the first corrected-alpha run collapsed distinct physical peaks
sharing a rounded word into one DreaMS contextual state. A later audit found
that the attempted physical-group correction still allowed a peak discarded by
DreaMS's top-100 truncation to borrow a nearby retained state. The replacement
requires exact retained peak identity and starts Hybrid and its feature pool
from an empty target, importing only the unchanged, hash-verified Tomotopy core
model. No earlier DreaMS pool, Hybrid checkpoint, Hybrid model, chemical
mixture, MAG result, or SOS result is reused. The raw-DreaMS worker remains
isolated from the legacy environment's duplicate OpenMP runtime; no unsafe
duplicate-runtime override is used.

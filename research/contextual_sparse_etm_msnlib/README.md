# Contextual Sparse ETM evidence

This directory contains the compact, sealed evidence used by the Contextual
Sparse ETM paper. It is generated from an external clean-room run rather than
used as a training cache.

The reproducible sequence is:

```bash
python -m scripts.run_contextual_sparse_etm_reproduction initialize --root RUN
python -m scripts.run_contextual_sparse_etm_reproduction run --root RUN
python -m scripts.package_contextual_sparse_etm_reproduction \
  --root RUN \
  --output research/contextual_sparse_etm_msnlib/evidence/20260901_clean_room
python -m scripts.generate_contextual_sparse_etm_report
```

The raw run remains outside the repository. The committed package contains the
source commit, acquisition and split identities, stage commands and hashes,
synthetic ablations, validation results, frozen-model test results, probability
audits, chemical-evaluation checks, and acceptance checks required by the
report. Host-specific absolute paths are removed before the package is sealed.

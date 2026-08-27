# Repro/code notes

This directory intentionally contains **compact, reviewable reference code**, not a second production implementation.

- `published_models_reference.py` contains the ETM, ECR and TopMost-style ECRTM equations used to anchor the research direction, plus the frozen `tau=0.30` theta sharpening operation.
- `SIMULATION_PROTOCOL.md` records the complete synthetic data design, truth definitions, metrics and negative-control conditions.
- The executable real-data candidate harness lives at `scripts/run_published_topic_models_msnlib.py`; it uses the repository's actual MSnLib split/vocabulary/SGNS preparation.

The earlier exploratory synthetic harness was deliberately not promoted into production-facing code. Its exact numerical outputs needed for the scientific conclusions are preserved under `../results/`, and the protocol is sufficient for an independent reimplementation if the screen must be repeated.

For the next phase, do not spend compute reproducing the synthetic screen unless a code audit finds a discrepancy. The priority is the locked real MSnLib validation described in `../NEXT_AGENT.md`.

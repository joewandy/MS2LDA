# Pooled Projected MS2LDA reference implementation

This directory preserves the production-facing code from the separate Neural MS2LDA simplification study so the real-data agent can compare it directly with the published-model candidates. It was originally developed against repository base commit `20de0e45aec25203e6bc38770a795b25cc18bff7`.

## Model

`PooledProjectedMS2LDA` keeps the existing 50-dimensional train-only token features (48 SGNS dimensions + fragment/loss indicators), learns a bias-free `50 -> 128` projection, forms a normalized count-weighted pooled spectrum embedding, and obtains `theta` by cosine similarity to one shared topic-prototype bank. The same projected words and topic prototypes define `beta`.

The supplied model uses the simplification study's 50/50 fragment/loss decoder. Important: the later ETM-specific channel-balance screen in `../CHANNEL_BALANCE_SIMULATION.md` found no reason to impose 50/50 on canonical fixed-SGNS ETM. Therefore treat channel balance as part of this pooled candidate's previously studied specification, not as a generic rule for all models.

## First configurations

- `protocol_minimum.json`: likelihood only (`mi_weight=0.0`). This is the scientific minimum and should be the primary pooled candidate.
- `protocol_mi005.json`: the same model plus weak assignment mutual information (`mi_weight=0.05`). This is secondary/diagnostic, not automatically preferred.

Do not tune these settings before the first locked MSnLib validation comparison.

## Integration boundary

The files here are reference overlay code rather than a replacement for the locked M1 implementation. The local agent should integrate/adapt them into its experiment branch while preserving the repository's existing split, vocabulary, SGNS, completion, MAG and SOS machinery. Do not change M1's committed evidence.

Required work for the real run:

1. wire the pooled model as a separate benchmark method;
2. preserve the existing artifact contract (`weights.pt`, `model.json`, `vocabulary.json` or an explicitly documented candidate equivalent);
3. infer validation theta with the one-pass pooled inference path;
4. send candidate beta/theta through the same existing completion and MAG/SOS evaluation used by M1;
5. save collapse, channel-mass, speed and memory diagnostics;
6. keep validation/test locking rules from `../LOCAL_EXPERIMENT_PLAN.md`.

## Synthetic evidence

The separate study found the likelihood-only pooled projected model strong on both a K=28 main simulation and a K=64 overcomplete stress test, with weak MI giving a modest additional recovery benefit in the K=64 stress. See `results/main_three_seed_summary.csv`, `results/scale_stress_summary.csv` and `results/decision.json`.

This evidence establishes that the architecture is worth a real-data comparison. It does not establish chemical superiority on MSnLib.

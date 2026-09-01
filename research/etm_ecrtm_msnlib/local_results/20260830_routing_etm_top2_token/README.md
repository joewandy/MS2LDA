# Zero-parameter top-2 token Routing ETM

Status: **completed negative synthetic triage**. Candidate test remained locked.

This package tests the only credible simplification left after selecting Routing
ETM: remove the leave-one-out spectrum context and its learned scalar while
keeping direct token-to-topic top-2 votes and alpha-entmax 1.5.

## Outcome

The parameter-free route is useful but insufficient. It substantially improves
over entmax-only ETM, showing that top-2 restriction itself repairs part of the
topic-starvation failure. Relative to the selected contextual model, however,
true-beta recovery fell by 0.088100, true-theta recovery by 0.103449, recovered
planted motifs from 10 to 6, active topics from 14 to 11 and unique top-1 topics
from 14 to 10. NLL was 1.0446% worse.

| metric | entmax ETM | top-2 token | top-2 context |
|---|---:|---:|---:|
| held-out NLL | 6.616673 | 6.344000 | **6.278416** |
| true-beta cosine | 0.250317 | 0.410354 | **0.498454** |
| true-theta cosine | 0.403055 | 0.661425 | **0.764875** |
| recovered planted motifs | 2 | 6 | **10** |
| active / unique top-1 topics | 5 / 4 | 11 / 10 | **14 / 14** |
| median effective / exact support | 1.816 / 3 | 2.005 / 4 | 1.971 / 4 |
| learned parameters | 2,167,400 | 2,167,400 | 2,167,401 |

The candidate is genuinely simpler: it has no context scalar and exactly the
same learned state and parameter count as balanced ETM. It remains finite,
sparse and non-catastrophically duplicated. Those properties are not enough to
replace a model whose purpose is broad, recoverable motif discovery.

## Decision

The experiment stopped at its first predeclared gate. Seeds 23/37, K=128 and
real MSnLib were not run. The selected one-scalar contextual Routing ETM remains
the simplest demonstrated formulation that solves both sparse per-spectrum
inference and global topic-inventory recovery.

The full protocol and stopping decision are in `EXPERIMENT_LOG.md`.
`config.json`, `synthetic_summary.csv` and `provenance.json` provide the compact
audit package; retained weights and arrays remain outside Git.

## Replay

From the repository root:

```bash
conda run --no-capture-output -n ms2lda-neural \
python -m scripts.run_routing_etm_campaign \
  --output-root /path/to/routing-etm-top2-token-campaign \
  --seed 11 --fitted-topics 36 --routing-variant top2_token \
  --theta-transform entmax15 --reconstruction-scaling raw_counts \
  --epochs 120 --batch-size 200 --device cuda --threads 6 \
  --training-documents 800 --validation-documents 160
```

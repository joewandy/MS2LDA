# Routing ETM with one positive-NPMI coherence term

Status: **completed negative synthetic triage**. Candidate test remained locked.

This directory records the bounded experiment predeclared in
`EXPERIMENT_LOG.md`. The candidate is the frozen Routing ETM plus one
train-derived positive-NPMI topic loss. No other architecture or objective
component changed, no parameters were added and the coefficient was not tuned.

## Outcome

The loss worked mechanically: train-graph coherence loss decreased from
5.526062 to 5.499502. It did not work scientifically. True-beta recovery changed
from 0.498454 to 0.491576, missing the predeclared requirement to improve by at
least 0.01. Held-out NLL worsened slightly from 6.278416 to 6.287552; true-theta
recovery, sparse support and topic inventory were effectively unchanged.

The experiment stopped at the first gate. Seeds 23/37, K=128 and real MSnLib
were not run. There was no coefficient search and no candidate-test access.

| metric | Routing ETM | + positive-NPMI |
|---|---:|---:|
| held-out NLL | 6.278416 | 6.287552 |
| true-beta cosine | **0.498454** | 0.491576 |
| true-theta cosine | 0.764875 | 0.765219 |
| train-graph coherence loss (lower is better) | 5.526062 | **5.499502** |
| active / unique top-1 topics | 14 / 14 | 14 / 14 |
| median effective / exact support | 1.971 / 4 | 1.955 / 4 |

## Interpretation

Positive-NPMI can improve the statistic it directly optimizes without improving
recovery of the planted motif-word distributions. In this formulation it adds a
training mechanism but no demonstrated scientific value. The simplest supported
paper model therefore remains Routing ETM without NPMI.

Exact predeclaration, result and stopping decision are in `EXPERIMENT_LOG.md`.
`config.json`, `synthetic_summary.csv` and `provenance.json` make the negative
result reviewable. Large weights, beta arrays and the graph remain outside Git.

## Replay

From the repository root:

```bash
conda run --no-capture-output -n ms2lda-neural \
python -m scripts.run_routing_etm_campaign \
  --output-root /path/to/routing-etm-npmi-campaign \
  --seed 11 --fitted-topics 36 --routing-variant top2_context \
  --theta-transform entmax15 --reconstruction-scaling raw_counts \
  --positive-npmi --epochs 120 --batch-size 200 --device cuda --threads 6 \
  --training-documents 800 --validation-documents 160
```

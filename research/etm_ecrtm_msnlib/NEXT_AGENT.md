# Routing-informed ETM next-agent handoff

## Publication boundary

Keep a published model as the paper-facing base. Do not propose the private M1
architecture as the publication model and do not start an M1 multiseed campaign.
M1 may be consulted only as donor/ablation evidence for a small mechanism that
can be explained independently inside a published base.

Candidate test remains locked. All work is synthetic or real validation until a
later independent review explicitly authorizes test.

## Current result

The new candidate is a balanced fixed-SGNS Embedded Topic Model with:

- unchanged ETM embedding decoder, likelihood, Gaussian variational family and
  analytic standard-normal KL;
- one learned leave-one-out context scalar;
- top-2 token evidence scored in ETM's own topic geometry and added as a bounded
  centered log offset to the posterior mean;
- published alpha-entmax 1.5 theta; and
- unchanged raw intensity pseudo-count reconstruction.

It has no nonlinear M1 router, document gate, Sinkhorn, NPMI, prototype
separation, alternating optimizer or schedule.

Synthetic K=36 seeds 11/23/37 and K=128 adjudication showed that routing prevents
the topic starvation caused by entmax alone while entmax supplies the exact
sparsity absent from routing+softmax. The K=128 candidate recovered all 18
planted motifs as unique top-1 topics.

Frozen real validation at K=1000 produced:

| metric | result | gate |
|---|---:|---:|
| optimized motifs | 803 | >=840 (fail) |
| evaluable motifs | 445 | >=388 (pass) |
| useful motifs | 289 | >=252 (pass) |
| mean SOS | 0.647153 | >=0.651498 (fail) |
| completion NLL | 9.542924 | <=9.422847 (fail) |
| median effective topics | 3.70 | diagnostic |
| median exact support | 6 | diagnostic |
| unique top-1 topics | 828 | diagnostic |
| corpus-effective topics | 538.40 | diagnostic |

The model is finite and has no catastrophic duplicate component. This is a
failed all-gates result, but it is the first ETM-family candidate here to repair
both per-spectrum diffuseness and global topic collapse while exceeding the
evaluable/useful targets.

Authoritative evidence is in
`local_results/20260830_routing_etm/README.md`, `EXPERIMENT_LOG.md`, and the CSV/
JSON artifacts beside them.

## Next bounded experiment

Test one intervention only: train-derived positive-NPMI topic-coherence
regularization added to the frozen routing-informed ETM.

Rationale: the remaining real deficits are optimized beta coverage and mean SOS,
with a small NLL excess. Positive-NPMI is directly connected to topic-word
coherence and is grounded in published topic-model evaluation/regularization.
It must be described independently rather than justified as “because M1 has it.”

Required sequence:

1. Predeclare the NPMI construction, coefficient-selection rule and stopping
   gates before training. Do not run an open-ended coefficient sweep.
2. Screen on the existing paired fragment/loss synthetic seeds 11/23/37 with
   K=36. Preserve routing+entmax recovery, sparse support and topic inventory.
3. Run the existing seed-11 K=128 stress only if all three K=36 seeds pass.
4. Promote at most one fixed coefficient/formulation to the same validation-only
   K=1000 MSnLib evaluation.
5. Require all original frozen gates, including NLL. Do not loosen them.

Do not add the document gate, Sinkhorn balancing, prototype separation,
alternating optimization, a routing-strength sweep or a temperature sweep in the
same campaign. If NPMI fails, stop and reassess rather than reconstructing M1.

## Shared immutable inputs

- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/zenodo/20179680`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/etm-pooled-validation-20260827-seed42`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/sparse-etm-campaign-20260830`
- `/home/joewandy/Work/data/MS2LDA-msnlib-validation/runs/routing-etm-campaign-20260830`

Use the existing `ms2lda-neural` Conda environment and `environment.yml`. Do not
redownload or rebuild MSnLib, SGNS, Spec2Vec, MAG or FAISS assets. Do not commit
weights, NumPy arrays, databases or indexes.

# Final neural-model selection conclusion

Status: validation-only architecture selection is closed. Candidate test data remain locked.

Evidence commits:

- research handoff: `9baec8aa62f684480eba35d4fc7f626c46f7b804`
- first real comparison: `ecb09251de94093e345584ed53f87ed799e88dc4`
- follow-up diagnosis: `c2c78cfb9cc7f35a1715312b9f7d4130ec60d4ba`

## Decision

M1 is the **least-complex model demonstrated to satisfy the complete real-data scientific contract**. This is more precise than calling it the mathematically simplest model. Simpler published or deterministic alternatives fitted the spectra, but none preserved the required combination of topic diversity, confident spectrum association, chemical breadth, SOS quality, and completion likelihood.

No alternative candidate is authorized for test.

## Locked validation result

| Model | Optimized | Evaluable | Useful | Mean SOS | Completion NLL | Outcome |
|---|---:|---:|---:|---:|---:|---|
| M1 | 884 | 408 | 265 | 0.658079 | 8.974140 | Pass |
| Canonical fixed-SGNS ETM | 609 | 130 | 79 | 0.638796 | 8.690730 | Fail chemistry |
| Pooled projected | 967 | 14 | 11 | 0.751621 | 8.385787 | Fail breadth/collapse |
| Pooled projected, theta tau 0.11 | 967 | 293 | 194 | 0.665436 | 8.994812 | Fail breadth/collapse |
| Pooled projected + MI 0.05 | 799 | 14 | 11 | 0.696540 | 8.393690 | Fail breadth/collapse |
| Fragment/loss-balanced ETM | 911 | 166 | 104 | 0.629819 | 8.766069 | Fail breadth/SOS |
| Canonical ECRTM | -- | -- | -- | -- | -- | Sinkhorn failed in epoch 22 |

The frozen candidate gates were optimized >=840, evaluable >=388, useful >=252, mean SOS >=0.651498, completion NLL <=9.422847, finite execution, and no catastrophic inventory collapse.

## What the alternatives established

### Canonical ETM

Fixed train-only SGNS made canonical ETM much healthier than a learned-embedding ETM in simulation, and its real completion NLL was good. It nevertheless learned only 609 MAG-optimizable motifs and produced diffuse spectrum mixtures. Post-hoc sharpening could not repair the topic-word deficit.

### Pooled projected model

The raw pooled model contained a useful specialized sub-inventory, but 614 of 1,000 topic-word distributions formed one near-exact cosine component. Those topics were nearly uniform over the vocabulary and never won a validation spectrum. Temperature calibration corrected probability scale and raised evaluable/useful counts to 293/194, but rank-preserving calibration could not manufacture missing topic identities.

This result also changes the benchmark contract: MAG optimization coverage must be interpreted together with topic use, beta concentration, unique top-1 topics, and duplicate-component size.

### Fragment/loss-balanced ETM

Channel balancing removed ETM's observed fragment/loss skew and increased optimized motifs from 609 to 911. It did not improve the full scientific outcome: evaluable/useful counts remained 166/104 and mean SOS fell. Channel imbalance was therefore a real symptom and contributor, but not the principal failure.

### ECRTM

ECRTM was a scientifically justified published anti-collapse comparator after the 614-topic pooled component was identified. The maintained ordinary-domain Sinkhorn formulation became progressively more expensive and failed the residual contract in epoch 22, including after an exact checkpoint resume. No partial or unconverged model was scored.

The appropriate conclusion is operational: this maintained ECRTM numerical path is unsuitable at K=1000 and V=21,233 on the tested hardware. It does not disprove every possible stabilized or GPU implementation of the mathematical model.

## Why M1's complexity is functional

The external model comparison exposes the same failure modes targeted by M1's internal mechanisms:

- token-level contextual routing and the document gate produce confident spectrum-topic assignments;
- Sinkhorn targets resist topic starvation;
- prototype separation resists duplicate components;
- positive-NPMI regularization supports coherent topic words;
- the fixed channel split avoids vocabulary-size-driven fragment/loss mass;
- shared prototypes couple the assignment and decoder geometries.

Existing within-M1 ablations show that removing the gate, Sinkhorn, NPMI, or separation substantially reduces useful motif inventory. The external comparisons now show that changing to a simpler model family does not make those problems disappear.

## Required diagnostics from now on

Every neural topic-model result must report, in addition to completion and MAG/SOS:

- median and mean effective topics per spectrum;
- corpus effective topic count;
- active topics above 0.0005 mean usage and at least 1/K usage;
- unique top-1 topics and topics never top-1;
- mean/median nearest-topic beta cosine and maximum pairwise cosine;
- connected duplicate-component summaries at beta cosine 0.95, 0.99, and 0.999;
- median beta effective-word count, maximum word probability, and top-20 mass;
- top-word uniqueness;
- per-topic fragment probability mass and extreme-skew fraction.

A large MAG-optimized count alone is not evidence of a usable topic inventory.

## Next compute

Architecture search is paused. The next campaign is M1 optimization-seed stability on the fixed seed-42 data split, vocabulary, token features, and evaluation contract. Test remains locked. The predeclared workflow is in `research/etm_ecrtm_msnlib/M1_MULTISEED_HANDOFF.md`.

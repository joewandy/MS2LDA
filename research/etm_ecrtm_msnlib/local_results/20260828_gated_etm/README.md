# Real-MSnLib detached-gate balanced ETM validation

This campaign is validation only. Candidate test theta, completion, chemistry,
MAG/SOS, and result artifacts were not opened, loaded, computed, or inspected.

## Current answer

The frozen `tau_g=1, gamma=1` model demonstrates a real but incomplete gate
effect. Relative to balanced ETM, it improves validation completion NLL,
evaluable/useful breadth, topic utilization, and beta redundancy, while leaving
the parameter count unchanged and preserving 50/50 fragment/loss mass. It does
not produce M1-like document assignments: median effective topics increase
from 46.72 to 64.79, high-confidence associations decrease, and chemistry gains
are far too small to pass the frozen breadth or mean-SOS gates.

A separately trained `tau_g=1, gamma=2` diagnostic improves chemistry to 220
evaluable / 138 useful motifs and mean SOS 0.645148, but it does not sharpen
documents: median effective topics increase again to 65.57 and only 22.55% of
spectra reach maximum theta 0.5. It remains far below the frozen breadth gates.
No temperature/gamma cross-product was run. Separation is not justified because
neither gated model has a beta pair at cosine 0.999. NPMI is not justified
because good document assignment has not been established.

## Validation headline

| Model | Optimized | Evaluable | Useful | Mean SOS | Median SOS | Completion NLL | Outcome |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| M1 locked reference | 884 | 408 | 265 | 0.658079 | 0.648864 | 8.974140 | Pass |
| Canonical fixed-SGNS ETM | 609 | 130 | 79 | 0.638796 | 0.631266 | 8.690730 | Fail breadth/SOS |
| Fragment/loss-balanced ETM | 911 | 166 | 104 | 0.629819 | 0.632929 | 8.766069 | Fail breadth/SOS |
| Balanced ETM + gate, tau 1 gamma 1 | 890 | 181 | 108 | 0.630221 | 0.629630 | 8.695312 | Fail breadth/SOS |
| Balanced ETM + gate, tau 1 gamma 2 | 889 | 220 | 138 | 0.645148 | 0.642500 | 8.686967 | Fail breadth/SOS |

## What the frozen gate buys

| Metric | Balanced ETM | Gate gamma 1 | Change |
| --- | ---: | ---: | ---: |
| Completion NLL | 8.766069 | 8.695312 | -0.070757 |
| Optimized motifs | 911 | 890 | -21 |
| Evaluable motifs | 166 | 181 | +15 |
| Useful motifs | 104 | 108 | +4 |
| Mean SOS | 0.629819 | 0.630221 | +0.000402 |
| Associated spectra | 1,240 | 1,020 | -220 |
| Associated molecules | 1,232 | 1,012 | -220 |
| Median effective topics/spectrum | 46.72 | 64.79 | +18.07 (worse) |
| Fraction max theta >= 0.5 | 0.3189 | 0.2623 | -0.0566 (worse) |
| Unique top-1 topics | 260 | 348 | +88 |
| Corpus effective topic count | 355.71 | 489.55 | +133.85 |
| Mean nearest beta cosine | 0.3996 | 0.3041 | -0.0955 (better) |
| Maximum beta cosine | 0.9958 | 0.9942 | -0.0017 (better) |

The result is therefore not a simple null: shared geometry improves beta
inventory and completion. But it broadens rather than sharpens the real
document mixtures, so the fixed `theta >= 0.5` chemistry bottleneck remains.

The stronger gamma improves chemistry relative to gamma 1 by 39 evaluable and
30 useful motifs, raises mean SOS by 0.014927, and lowers NLL by 0.008345. It
also raises unique top-1 topics from 348 to 471 and corpus effective topics from
489.55 to 589.36. Those gains come without M1-like assignment: associated
spectra fall from 1,020 to 877, median maximum theta falls from 0.3232 to 0.3024,
and median effective topics rise from 64.79 to 65.57.

## Frozen-gate result

Both gated candidates pass optimized coverage, completion NLL, finite/stable,
and no-catastrophic-duplicate gates. The stronger gamma fails evaluable motifs
(220 < 388), useful motifs (138 < 252), and mean SOS
(0.645148 < 0.651498). It is not near passing and must not advance to test.

## Explicit scientific questions

1. **Does the gate materially improve balanced ETM?** Yes for NLL, topic use,
   beta redundancy, and—with stronger gamma—some chemistry; not enough for M1.
2. **Does it increase evaluable/useful breadth?** Gamma 1 raises 166/104 to
   181/108; gamma 2 reaches 220/138, still far below 388/252.
3. **Does it improve or damage mean SOS?** Gamma 1 is essentially unchanged;
   gamma 2 reaches 0.645148, better than 0.629819 but below the frozen gate.
4. **Does it reduce theta diffuseness?** No. Gamma 2 has 65.57 median effective
   topics versus 46.72 for balanced ETM, and only 22.55% reach theta 0.5.
5. **Does it reduce duplication without separation?** Yes. Neither gated model
   has a pair above cosine 0.999; gamma 2 maximum cosine is 0.99354.
6. **Does it preserve the NLL gate?** Yes; gamma 2 improves NLL to 8.686967.
7. **Does either gated model pass all frozen validation gates?** No.
8. **What single failure remains?** M1-level high-confidence chemistry breadth
   remains absent; the immediate measured mechanism is diffuse theta.
9. **Did the targeted gamma-2 experiment fix it?** No. It materially improves
   chemistry but remains 168 evaluable and 114 useful motifs below the gates.
10. **Is the model still recognizably ETM?** Yes. It retains the original ETM
    variational encoder, logistic-normal theta, KL, embedding decoder, optimizer,
    and parameter count, adding only channel normalization and a parameter-free
    detached geometry gate.
11. **Is it scientifically preferable to M1?** No. It is simpler to explain,
    but M1 alone achieves the required chemistry and substantially sparser
    document assignments.
12. **Should it advance to test?** No candidate is authorized for test in this
    validation-only campaign.

Exact expanded diagnostics are in `comparison.csv` and the per-model artifact
directories. Experiment rationale and decisions are in `EXPERIMENT_LOG.md`;
large local artifacts and SHA-256 hashes are in `provenance.json`.

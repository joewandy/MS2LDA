# Neural MS2LDA experimental ledger

This ledger preserves the negative and comparative findings that motivated the
supported K=500 ERNTM checkpoint. Rejected implementations remain recoverable
from the safety bundle and archive branches; they are deliberately absent from
active `main`.

| Method family | Outcome | Decision |
|---|---|---|
| Random-start amortized LDA, K=200 | Used 1/200 topics; NLL 7.919 versus 4.952 for its Tomotopy reference; top-20 diversity 0.061. | Rejected: severe component collapse. |
| NMF beta initialization | Retained 71/200 topics but remained substantially weaker than the reference. | Rejected: initialization alone did not solve discovery. |
| Fully warm-started beta and encoder | Retained all 200 topics but joint training did not improve the non-neural initialization. | Rejected: not independent neural discovery. |
| Semi-amortized HybridLDA | Prevented collapse and accelerated inference, but retained classical expected-count/VB updates and was less stable across runs. | Useful hybrid result, not fully neural. |
| First neural-assignment K=1000 | Stable and fast, with 257 active topics, but diversity 0.3646 and mean NPMI -0.4962; chemical quality was credible only for a limited subset. | Rejected as final form: semantic duplication. |
| ECR capacity arms | Improved some diversity points but did not move the complete likelihood, inventory, and chemistry frontier beyond ERNTM K=500. | Rejected from the supported model; evidence retained in `capacity_screen.csv`. |
| Baseline capacity arms | Increasing K enlarged the mass-carrying inventory while reducing global top-word diversity. | Diagnostic evidence that nominal K is not usable motif count. |
| ERNTM K=300 | High diversity but a smaller non-redundant inventory than K=500. | Not selected. |
| ERNTM K=500 | Best defensible compromise among diversity, usable inventory, likelihood, speed, and chemical evidence. | Selected checkpoint. |
| Balanced-gated neural K=1000 | Soft fragment--loss balancing plus a fixed document gate produced 148 useful validation motifs and 211 test motifs, versus Tomotopy's 138 and 186; neural MAG coverage remained lower. | Selected seed-42 research checkpoint; multi-seed confirmation pending. |

The decisive structural finding is that balanced routing and recycling prevent
gross activity collapse, while ERNTM prototype separation reduces duplication.
At K=1000, document gating supplies confident mixtures and soft fragment--loss
balancing makes far more of that inventory chemically useful. The neural model
still has lower broad MAG coverage than Tomotopy. Its research value is fully
neural unsupervised discovery plus one-pass inference; this single-seed result
does not establish production replacement.

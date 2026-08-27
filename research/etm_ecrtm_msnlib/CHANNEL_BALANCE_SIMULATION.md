# ETM fragment/loss channel-balance simulation screen

Status: **mechanism screen, not a converged model-selection result**.

Date: 26 August 2026.

## Question

Does forcing every ETM topic to allocate exactly 50% probability mass to fragment words and 50% to neutral-loss words improve a faithful fixed-SGNS ETM on short sparse MS/MS-like spectra?

The comparison changes only decoder normalization:

- **global softmax**: canonical ETM topic-word softmax over the full vocabulary;
- **channel balanced**: the same ETM logits, but independent softmaxes over fragment and loss vocabularies, each multiplied by 0.5.

Within every paired run the two models use identical initial weights, training row order, stochastic-posterior RNG seed, data, fixed SGNS vectors, optimizer and epoch count.

## Data regimes

The existing short-sparse paired-peak simulator was reused. K=36 is fitted to 18 planted motifs. The vocabulary is train-only and SGNS is train-only. For each seed the same SGNS embedding is held fixed across the three channel-mass regimes to isolate decoder/channel effects.

Three regimes were evaluated by rescaling the observed fragment/loss pseudo-count mass while preserving physical spectra and average total mass:

1. `balanced_truth`: fragment scale 1.0, loss scale 1.0; true motif fragment mass ~0.50.
2. `fragment_heavy_70_30`: fragment scale 1.4, loss scale 0.6; true motif fragment mass ~0.70.
3. `loss_heavy_30_70`: fragment scale 0.6, loss scale 1.4; true motif fragment mass ~0.30.

The vocabulary itself is almost exactly balanced in the representative seed: 925 fragment words and 934 loss words.

## Main screen

Three seeds: 11, 23, 37. ETM equations/ELBO are unchanged, but a 200-unit encoder and 15 epochs are used so the full paired mechanism screen is locally tractable. This is intentionally a direction-of-effect screen, not a replacement for the earlier 800-unit/120-epoch ETM benchmark.

| regime | decoder | NLL | true-beta cosine | true-theta cosine | active topics (>0.5% usage) | nearest-topic redundancy | learned fragment mass |
|---|---|---:|---:|---:|---:|---:|---:|
| ~50/50 truth | global | 7.2738 | 0.1716 | 0.4416 | 31.7 | 0.9887 | 0.4935 |
| ~50/50 truth | forced 50/50 | 7.2735 | 0.1716 | 0.4417 | 32.7 | 0.9887 | 0.5000 |
| ~70/30 truth | global | 7.2797 | 0.1614 | 0.4479 | 23.3 | 0.9887 | 0.4957 |
| ~70/30 truth | forced 50/50 | 7.2777 | 0.1627 | 0.4501 | 20.0 | 0.9896 | 0.5000 |
| ~30/70 truth | global | 7.2547 | 0.1650 | 0.4739 | 31.3 | 0.9888 | 0.4912 |
| ~30/70 truth | forced 50/50 | 7.2613 | 0.1637 | 0.4919 | 31.7 | 0.9890 | 0.5000 |

Paired mean change (balanced minus global):

- **50/50 truth:** NLL -0.0003, beta cosine +0.00007, theta cosine +0.00014, redundancy -0.000005. Essentially no effect.
- **70/30 truth:** NLL -0.0019, beta cosine +0.0013, theta cosine +0.0022, but ~3.3 fewer active topics and redundancy +0.00089. Mixed/negligible.
- **30/70 truth:** NLL +0.0066, beta cosine -0.0013, theta cosine +0.0180, redundancy +0.00018. Mixed, with worse NLL/beta but somewhat better theta.

No regime shows a consistent multi-metric advantage from forcing 50/50.

## Original-size encoder check

Seed 11 was repeated at the original ETM encoder width (800 units), still at 15 epochs. The direction is the same:

- **50/50 truth:** essentially identical NLL/beta/theta/redundancy.
- **70/30 truth:** forced balance slightly improves NLL but slightly worsens beta/theta and reduces active topics (7 vs 13).
- **30/70 truth:** forced balance worsens NLL, beta and theta slightly.

This check makes it unlikely that the neutral/mixed finding is merely caused by the 200-unit screening encoder.

## Unexpected diagnostic

The canonical global-softmax ETM stays close to 50% fragment mass even when the observed/true motif mass is deliberately moved to ~70% or ~30% during this early screen. This means the dramatic 1.9%-98.1% channel-mass pathology seen in the other pooled-model study does **not** automatically transfer to fixed-SGNS ETM.

Possible reasons include the almost-equal fragment/loss vocabulary sizes and the fixed SGNS geometry. Longer converged training might eventually move the global ETM channel mass further toward the asymmetric count truth; the attempted 120-epoch local paired confirmation was operationally too slow and is not claimed here.

## Decision

**Do not add fixed 50/50 channel balancing to ETM by default based on simulation.**

The earlier pooled-model result remains valid for that model family, but this paired ETM screen says the correction is not generically necessary for ETM. For the real MSnLib comparison:

1. run plain fixed-SGNS ETM first;
2. record each learned topic's fragment-mass distribution;
3. only run ETM + 50/50 balancing if real ETM actually exhibits a material channel-skew pathology;
4. judge any balanced variant on MAG/SOS chemistry, motif inventory and completion NLL, not channel mass alone.

This is a useful negative result: it prevents carrying an MS-specific constraint into the paper merely because another model family needed it.

## Limitations

- Main paired screen is 15 epochs with a 200-unit encoder; seed-11 original-width checks are also 15 epochs.
- K=36/V~2k simulation is much smaller than K=1000/V~21k MSnLib.
- The 70/30 and 30/70 regimes are controlled count-mass perturbations, not mechanistic simulations of instrument-specific fragment/loss missingness.
- Real chemical validation remains decisive.

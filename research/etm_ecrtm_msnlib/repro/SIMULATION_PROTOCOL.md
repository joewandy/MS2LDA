# Synthetic MS/MS reproduction protocol

This note records the synthetic design used to screen ETM/ECRTM before real MSnLib validation. It is intended to make the numerical CSVs auditable without making the exploratory simulation harness part of production MS2LDA.

## Fixed study design

Seeds: `11, 23, 37`.

Per seed:

- true motifs: 18;
- training spectra: 800;
- validation spectra: 160;
- test spectra: 160;
- physical peaks per spectrum: integer uniform 18-42;
- active motifs per spectrum: 1-3 with probabilities approximately 0.46/0.39/0.15;
- motif prevalence: long-tailed Zipf-like prior with exponent about 0.72;
- training vocabulary minimum document frequency: 3;
- SGNS embeddings: train only, 48 dimensions for the final ETM/ECRTM screen;
- completion split: whole physical peaks, approximately 50/50 observed/completion;
- fragment and neutral loss from the same peak always remain on the same side of the completion split.

## Spectral-token realism

Each planted motif contains recurrent fragment and neutral-loss anchors. A minority of anchors are deliberately shared across motifs, making some words ambiguous. Spectra additionally contain common background anchors and low-intensity random noise.

A physical peak is generated either from a motif fragment anchor, a motif neutral-loss anchor, a background anchor or random noise. Given precursor mass, a fragment and its complementary neutral loss are generated as a pair where physically valid.

Masses are rounded to two decimals and written as the current MS2LDA-style tokens:

- `frag@<m/z>`;
- `loss@<neutral loss>`.

Raw peak intensities are log-normal and heterogeneous. Intensities are normalized to the maximum retained peak in the spectrum. The current MS2LDA pseudo-count representation is then applied:

`count = max(1, round(100 * normalized_intensity))`.

Consequently the simulator reproduces the important short/sparse-vs-count-mass mismatch:

- median physical peaks about 29;
- median in-vocabulary nonzero fragment/loss terms about 25-26;
- median total pseudo-count mass roughly 1,369-1,449 depending on seed.

The last point is important for variational neural topic models because reconstruction scales with pseudo-count mass while KL does not.

## Train-only vocabulary and OOV handling

The vocabulary is built only from training spectra. Validation/test fragment/loss tokens absent from the training vocabulary are OOV and excluded from the in-vocabulary completion likelihood while their mass is recorded separately.

This mirrors the real benchmark's evidence boundary.

## True beta

For each planted motif, the truth-known topic-word distribution is formed from the count-weighted motif-labelled fragment/loss contributions observed in the training spectra, restricted to the training vocabulary, then row-normalized.

This gives an exact target for learned motif recovery that does not exist on real MSnLib.

## True theta

For each spectrum, the truth-known motif mixture is the normalized count-weighted mass attributable to the planted active motifs.

Background/noise evidence is not assigned to a planted motif.

## ETM conditions

Two legitimate modes of the published ETM model were tested:

1. fixed pretrained SGNS word embeddings;
2. jointly learned word embeddings.

The original ETM ELBO is retained. The final comparison used 120 epochs.

Two topic-count conditions were used:

- `K=18`: correctly specified;
- `K=36`: deliberately overcomplete, twice the planted motif count, to expose component/topic collapse.

## ECRTM condition

ECRTM follows the published/TopMost architecture and initializes word embeddings from the same train-only SGNS vectors.

The three-seed screen used:

- encoder units: 200;
- beta temperature: 0.2;
- ECR weight: 100 (maintained TopMost default used in this screen);
- Sinkhorn alpha: 20;
- bounded Sinkhorn maximum iterations: 50;
- training epochs: 40;
- batch size: 200;
- learning rate: 0.002.

The 50-step cap is a numerical approximation. It was not assumed blindly: one representative K=36, seed-11 experiment was repeated with the standalone implementation's exact 1000-step cap and produced nearly identical scientific metrics. Both rows are preserved under `results/`.

## Metrics

### Held-out NLL

Infer theta from the observed half and score the completion half under the model's exact decoder.

For ETM:

`p(w|d) = theta_d @ beta`.

For ECRTM, use the model's actual decoder:

`softmax(decoder_bn(theta @ beta_internal))`.

### True-beta recovery

Compute cosine similarity between every planted motif beta and every learned topic row, then perform one-to-one Hungarian matching to maximize total cosine. Report mean matched cosine.

For ECRTM, topic interpretation uses its published topic/word ECR geometry. Row normalization does not change within-row rankings or cosine similarity.

### True-theta recovery

Use the same Hungarian true-beta matching to align learned topic columns to planted motif columns, then compute per-spectrum cosine between true and aligned learned theta. Report mean cosine.

### Effective topics per spectrum

For normalized theta:

`effective_topics(d) = exp(-sum_k theta[d,k] * log(theta[d,k]))`.

Report the median over spectra.

### Corpus effective topics

Compute mean topic usage over spectra and exponentiate its entropy.

### Material activity

The preserved CSVs include activity under mean corpus usage `> 0.005` for the synthetic K<=36 screen. The real MSnLib runner additionally reports usage relative to `1/K` and `>0.0005` because K=1000 requires more appropriate diagnostics.

### Topic redundancy

Row-normalize learned topic-word vectors for cosine, calculate pairwise learned-topic cosine, exclude the diagonal, and report mean nearest-topic cosine. Larger values indicate more duplicated topics.

## Theta temperature experiment

After training ECRTM, deterministic theta is recalibrated without changing model weights or beta:

`theta_tau = softmax(log(theta) / tau)`.

The sweep was performed at K=36.

Tau `0.30` was chosen on seed 11 as the sparse-mixture compromise and then frozen unchanged on seeds 23 and 37. The three-seed summary is in `results/theta_temperature_three_seed_summary.csv`.

This frozen value may be evaluated on real MSnLib validation as a predeclared calibration. Do not re-select tau on the MSnLib test split.

## Negative controls preserved

`results/direct_etm_ecr_negative_control_seed11_K36.csv` shows why a custom ETM+ECR splice is not recommended.

`results/ecrtm_alpha01_seed11_K36.csv` shows why simply changing the symmetric Dirichlet concentration to 0.1 did not solve diffuse ECRTM theta in this count representation.

`results/ecrtm_exact_sinkhorn1000_weight250_seed11_K36_40epoch.csv` and `results/ecrtm_sinkhorn50_weight250_seed11_K36_40epoch.csv` record the numerical approximation spot check.

## Reproducibility expectation

A future agent does not need to reproduce the synthetic screen before running MSnLib unless code changes cast doubt on these results. If the screen is repeated, use the fixed design above and preserve the three seeds and K=18/36 conditions so comparisons remain interpretable.

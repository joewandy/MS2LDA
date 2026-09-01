# Reference Neural Sinkhorn Topic Model campaign

## Decision

An official reference implementation exists and was tested. NSTM does **not**
advance to real MSnLib validation from this synthetic screen.

The primary source is the authors' MIT-licensed TensorFlow implementation of
Zhao et al.'s [Neural Topic Model via Optimal
Transport](https://arxiv.org/abs/2008.13537), pinned at commit
`610d1604d5467289028714ed0ce684dfb5ef8a7b`. A local PyTorch 2.10 port preserves
the published encoder, fixed pretrained word embeddings, cosine ground cost,
Sinkhorn objective and virtual reconstruction decoder. Its Sinkhorn costs agree
with the maintained [TopMost](https://github.com/bobxwu/TopMost) PyTorch port at
commit `ef24433859b2e283959ddef7f95020a40abb104f` to a maximum absolute
difference of `7.45e-08` on matched tensors.

The screen found one promising property: the released-code form recovered
planted document mixtures and top-word sets better than the balanced ETM
control. It nevertheless failed as a topic model for this domain because its
topic-word distributions were nearly uniform and mutually redundant, its
completion likelihood was worse, and its document mixtures became much more
diffuse as K increased.

## Reference ambiguity tested explicitly

The paper defines `x_tilde = counts / sum(counts)` for the encoder, transport
marginal and expected likelihood. The released TensorFlow script instead:

1. feeds raw counts to the encoder;
2. uses `softmax(raw_counts)` as the word transport marginal; and
3. weights reconstruction by raw counts.

This distinction is easy to overlook when inspecting the authors' low-integer
TMN data (maximum count 6, median 16 nonzero words), but is plainly consequential
for MS/MS intensity pseudo-counts (maximum 131, median real training mass 526).
Both interpretations were tested without tuning.

## Synthetic K=36 result

All rows use the same seed-11 truth-known MS/MS simulator: 18 planted motifs,
1--3 motifs per spectrum, paired fragments/losses, 800 training and 160
validation spectra, and the same fixed train-only 48D SGNS embeddings. NSTM
uses its official 200-unit encoder, Adam 0.001, 50 epochs, epsilon 0.07 and
Sinkhorn alpha 20.

| Model | NLL | beta cosine | beta top-20 Jaccard | theta cosine | top-motif accuracy | median effective topics | active / unique top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced ETM control | **6.4733** | **0.3114** | 0.1587 | 0.4968 | 0.4750 | **3.25** | 36 / 7 |
| NSTM, paper L1 | 7.4818 | 0.1611 | 0.1191 | 0.5001 | 0.4000 | 22.91 | 36 / 32 |
| NSTM, released code | 7.3366 | 0.1866 | **0.2536** | **0.7476** | **0.7875** | 18.60 | 36 / 31 |

The paper-normalized objective was genuinely balanced at epoch 50:
`0.07 * reconstruction = 0.5223` and Sinkhorn loss `0.4921`. It still learned
weak planted beta, dense theta and worse completion.

In the released-code run, `0.07 * reconstruction = 725.5` while Sinkhorn loss
was only `0.8196`. The pseudo-count reconstruction term was about 885 times
larger, so the supposed transport model was trained almost entirely by the
count-scale reconstruction objective. Its mean nearest-topic beta cosine was
`0.9952`, despite improved top-word overlap and theta recovery.

## K=128 stress

Only the more promising released-code form received the predeclared high-K
stress. Changing K was the sole intervention.

| Model | NLL | beta cosine | beta top-20 Jaccard | theta cosine | median effective topics | active / unique top-1 | mean nearest beta cosine |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced ETM control | **6.4175** | **0.3696** | 0.2186 | 0.5782 | **3.67** | 128 / 7 | **0.9637** |
| NSTM, released code | 7.3488 | 0.1914 | **0.3393** | **0.7424** | 63.80 | 128 / 63 | 0.9962 |

The favorable theta/top-word signal survived, but median effective topics rose
from 18.6 at K=36 to 63.8 at K=128. All 128 topics joined one duplicate
component at beta cosine 0.99, top-word uniqueness fell to 0.1375, and the
virtual topic distributions retained roughly 1,793 effective words out of a
1,833-word vocabulary. This is the opposite of the required high-K behavior.

## Promotion outcome

Real K=1000 validation was not run. The frozen promotion rule requires
preserved topic recovery, controlled diffuseness, healthy distinct topic
inventory and competitive completion before consuming real-data evaluation.
NSTM failed those conditions at K=36 and the failure worsened at K=128.

This is useful negative evidence rather than an implementation failure:

- focused NSTM tests pass;
- gradients, losses and Sinkhorn marginals remained finite;
- the PyTorch Sinkhorn calculation matches the maintained reference port;
- all three GPU runs completed with the official optimization settings;
- no candidate test artifact was accessed.

The released NSTM encoder appears capable of extracting a document-partition
signal, but NSTM alone is not a viable MS2LDA base in its published form. Using
that signal would require a new decoder or strong sparsity/inventory machinery,
which would be a new hybrid rather than a faithful NSTM baseline.

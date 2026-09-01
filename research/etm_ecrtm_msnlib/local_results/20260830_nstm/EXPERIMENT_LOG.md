# Reference NSTM experiment log

Evidence boundary: truth-known synthetic MS/MS first. Real MSnLib validation is
allowed only after a frozen synthetic promotion decision. Candidate test theta,
completion, MAG, SOS and result artifacts must not be opened or computed.

## Reference implementation decision

- **Primary source:** He Zhao's MIT-licensed author repository for *Neural Topic
  Model via Optimal Transport*, commit
  `610d1604d5467289028714ed0ce684dfb5ef8a7b`.
- **Modern cross-check:** TopMost's PyTorch NSTM at commit
  `ef24433859b2e283959ddef7f95020a40abb104f`.
- **Implementation:** a local PyTorch 2.10 port of the author equations. Its
  Sinkhorn output was checked numerically against TopMost on the same tensors;
  maximum absolute cost difference was `7.45e-08`.
- **Measured discrepancy:** the paper defines L1-normalized word distributions,
  while the released TensorFlow script uses raw counts for the encoder and
  reconstruction and `softmax(raw_counts)` for the transport marginal. The
  author TMN data have maximum count 6 and median 16 nonzero words; real MSnLib
  has maximum pseudo-count 131, median mass 526 and median 37 nonzero words.
  Both behaviours are therefore named and tested rather than silently mixed.

## N0-paper: faithful paper-normalized NSTM

- **Hypothesis:** NSTM's direct transport between a short spectrum's word
  distribution and its topic distribution, using fixed train-only SGNS word
  embeddings, will recover planted motifs and document mixtures better than the
  balanced ETM seed-11 control without M1 routing, gates, NPMI or separation.
- **Exact model:** one 200-unit ReLU encoder, dropout probability 0.25, batch
  normalization, dense softmax theta, learned topic embeddings, fixed 48D
  train-only SGNS, cosine ground cost, differentiable reference Sinkhorn loss,
  and the virtual reconstruction decoder. L1-normalized counts are used exactly
  as specified by the paper.
- **Config:** 18 planted motifs; fitted K=36; seed 11; 800 training and 160
  validation spectra; Adam 0.001; batch 200; 50 epochs; epsilon 0.07; Sinkhorn
  alpha 20, maximum 1000 iterations and tolerance 0.005; CUDA.
- **Stopping rule:** stop immediately on non-finite objective/gradient or
  failure of the Sinkhorn marginal contract. Advance to seeds 23/37 and K=128
  only if beta/theta recovery and inventory are at least competitive with the
  paired balanced-ETM control and no collapse occurs.

## N0-code: released-code compatibility check

- **Hypothesis:** if the exact released count handling fails while N0-paper is
  healthy, the difference identifies a pseudo-count scaling incompatibility,
  not evidence against the published NSTM formulation itself.
- **Exact change from N0-paper:** raw counts enter the encoder and
  reconstruction term, and `softmax(raw_counts)` supplies the word transport
  marginal, matching the released TensorFlow and TopMost code. No model or
  optimizer setting changes.
- **Config:** seed 11, K=36 and all other settings paired with N0-paper.
- **Stopping rule:** this is a single compatibility run, not a tunable candidate.
  Do not promote released-code count handling merely because one metric wins.

## Seed-11 K=36 result

Both reference interpretations completed 50 CUDA epochs with finite objectives,
gradients and Sinkhorn marginals.

| Model | NLL | beta cosine | beta top-20 Jaccard | theta cosine | top-motif accuracy | median effective topics | unique top-1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| balanced ETM control | 6.4733 | 0.3114 | 0.1587 | 0.4968 | 0.4750 | 3.25 | 7 |
| NSTM paper L1 | 7.4818 | 0.1611 | 0.1191 | 0.5001 | 0.4000 | 22.91 | 32 |
| NSTM released code | 7.3366 | 0.1866 | 0.2536 | 0.7476 | 0.7875 | 18.60 | 31 |

The paper-normalized model balanced the final weighted reconstruction and
Sinkhorn terms (`0.07 * 7.461 = 0.522` versus `0.492`) but did not recover the
planted beta and remained diffuse. The released-code model improved theta
recovery, top-motif accuracy and top-word overlap, but raw pseudo-count
reconstruction dominated its final Sinkhorn term (`0.07 * 10364.1 = 725.5`
versus `0.820`). Its virtual topic-word distributions were also nearly
identical: mean nearest-topic beta cosine `0.9952`.

The primary N0-paper candidate is rejected. N0-code is not promoted yet, but
its strong theta/top-word signal justifies one already-bounded overcomplete
stress before closing the model family.

## N1: released-code K=128 stress

- **Hypothesis:** if the released reference genuinely exploits short-document
  transport rather than the K=36 setting, its theta recovery and top-word
  signal should survive a rise from 2x to about 7x the 18 planted motifs without
  effective theta scaling toward K or catastrophic topic duplication.
- **Exact change:** fitted topic count only, from K=36 to K=128. Retain seed 11,
  data, SGNS, released-code input handling and every official hyperparameter.
- **Config:** 50 epochs, batch 200, CUDA; otherwise identical to N0-code.
- **Stopping rule:** reject real promotion if true-theta recovery falls below
  the K=128 balanced-ETM control (`0.5782`), median effective topics scales
  substantially with K, beta similarity worsens from its already high K=36
  value, or completion/recovery remains clearly inferior overall. Run no
  K=256/K=1000 synthetic rescue.

## K=128 result and final promotion decision

The NSTM stress completed 50 CUDA epochs with finite losses, gradients and
Sinkhorn marginals. True-theta cosine remained strong at `0.7424`, top-motif
accuracy was `0.8063`, and beta top-20 Jaccard increased to `0.3393`. Those
signals exceed the K=128 ETM control (`0.5782`, `0.5438`, `0.2186`).

The stopping rule nevertheless fired on every structural quantity:

- median effective topics rose from `18.60 / 36` to `63.80 / 128`;
- true-beta cosine was only `0.1914` versus ETM's `0.3696`;
- completion NLL was `7.3488` versus ETM's `6.4175`;
- mean nearest-topic beta cosine worsened to `0.9962`;
- all 128 topics formed one beta duplicate component at cosine `0.99`;
- median beta effective words were `1,793 / 1,833`;
- top-word uniqueness fell to `0.1375`.

The released-code objective also remained reconstruction dominated:
`0.07 * 10331.6 = 723.2` versus Sinkhorn `0.7759` at epoch 50.

**Decision:** reject both faithful NSTM interpretations. Do not run seeds 23/37,
synthetic K=256/1000, or real MSnLib K=1000. The primary model failed at K=36;
the compatibility form retained a useful document-partition signal but failed
the predeclared high-K sparsity, beta recovery, completion and distinct-topic
inventory gates. No candidate test artifact was accessed.

# Zero-parameter top-2 token Routing ETM experiment log

Status: **completed negative synthetic triage**. The protocol was committed at
`18290f2614c928240b5aeccd33e101bba604e81b` before implementation or training.
Validation only; candidate test remains locked.

## Experiment 1: remove contextual routing

- **Question:** can the final Routing ETM be simplified by removing the
  leave-one-out spectrum context and its single learned scalar without losing
  the recovery, inventory, sparsity or real discovery result?
- **Published base:** unchanged balanced Embedded Topic Model with fixed
  train-only SGNS, Gaussian amortized inference, standard-normal KL, raw-count
  multinomial reconstruction and alpha-entmax 1.5.
- **Only change relative to the selected model:** each observed word is compared
  directly with the ETM topic coordinates and retains its two strongest topic
  matches. Count-weighted token votes are aggregated and added as the same
  centred bounded log-posterior offset. No leave-one-out context is computed and
  no context scalar is learned.
- **Complexity requirement:** the candidate must have exactly the same learned
  parameter count and state keys as the corresponding balanced ETM control. It
  may not introduce a fixed context coefficient, temperature/strength sweep or
  replacement mechanism.
- **Unchanged components:** top-2 evidence, the `1/K` posterior pseudocount,
  fixed routing temperature and strength 1.0, alpha-entmax 1.5, raw-count
  reconstruction, optimizer, decoder, data, seeds and evaluation.
- **Excluded changes:** no document gate, Sinkhorn target, NPMI loss, prototype
  separation, alternating optimizer, new prior, reconstruction rescaling or
  other M1 component.

## Synthetic protocol and gates

- **Controls:** reuse the exact saved top-2-context-plus-entmax and entmax-only
  artifacts with identical synthetic data, initialization, batch order and
  evaluation.
- **Initial triage:** seed 11, K=36, 120 epochs, 800/160 train/validation spectra.
- **Triage gate:** finite loss and gradients; zero added parameters; no
  catastrophic duplicate component; true-beta and true-theta cosine each no
  more than 0.02 below the contextual model; held-out NLL no more than 1% worse;
  at least 10 planted motifs recovered at cosine 0.50; active and unique-top-1
  topics each no more than two below the contextual model; median effective
  topics no greater than 3 and median exact support no greater than 6.
- **Multi-seed gate:** only after triage passes, repeat K=36 at seeds 23 and 37.
  On every seed beta and theta may fall by at most 0.03, NLL may worsen by at
  most 1%, active and unique-top-1 counts may fall by at most two, median
  effective topics must remain no greater than 3, execution must remain finite
  and no catastrophic component may appear. Across all three seeds, mean beta
  and theta may fall by at most 0.02.
- **High-K gate:** only after the multi-seed gate passes, run seed 11 at K=128.
  It must recover all 18 planted motifs as distinct top-1 topics, retain at least
  18 active topics, keep beta and theta within 0.03 of the contextual model,
  keep NLL within 1%, median effective topics no greater than 3, finite
  execution and no catastrophic duplicate component.
- **Stopping rule:** stop immediately at the first failed stage. Do not add back
  a coefficient, widen a threshold or introduce another mechanism.

## Real-validation promotion rule

At most this zero-parameter formulation may advance, using the existing frozen
seed-42 train/validation split, K=1000, training seed 7043 and unchanged MAG/SOS
evaluation. Candidate test must remain inaccessible.

The simpler model replaces contextual Routing ETM only if it has exactly the
balanced-ETM parameter count and retains all of the following:

- at least 787 optimized, 439 evaluable and 274 useful motifs;
- mean SOS at least 0.637558;
- completion NLL no more than 1% above 9.542924;
- median effective topics no greater than 4 and median exact support no greater
  than 8;
- at least 800 unique top-1 validation topics;
- finite execution and no catastrophic duplicate component.

These thresholds preserve the observed three-seed Routing ETM ranges rather
than merely requiring improvement over plain ETM. If any condition fails, keep
the contextual one-scalar model as the simplest supported formulation.

## Result

The seed-11 K=36 candidate completed with finite loss and gradients and exactly
the balanced-ETM parameter count: 2,167,400, one fewer than contextual Routing
ETM. It remained sparse and avoided catastrophic duplication. Top-2 restriction
alone also improved substantially over entmax-only ETM, but it did not retain
the selected contextual model's recovery or inventory.

| metric | entmax ETM | top-2 token | top-2 context | token minus context |
|---|---:|---:|---:|---:|
| held-out NLL | 6.616673 | 6.344000 | 6.278416 | +0.065584 |
| true-beta cosine | 0.250317 | 0.410354 | 0.498454 | -0.088100 |
| true-theta cosine | 0.403055 | 0.661425 | 0.764875 | -0.103449 |
| planted motifs recovered at cosine >=0.50 | 2 | 6 | 10 | -4 |
| active topics above 0.5% | 5 | 11 | 14 | -3 |
| unique top-1 topics | 4 | 10 | 14 | -4 |
| median effective topics | 1.816 | 2.005 | 1.971 | +0.034 |
| median exact support | 3 | 4 | 4 | 0 |
| mean nearest beta cosine | 0.930121 | 0.807493 | 0.740197 | +0.067295 |
| learned parameters | 2,167,400 | 2,167,400 | 2,167,401 | -1 |

The candidate missed every primary non-inferiority gate: beta and theta losses
were much larger than 0.02, NLL was 1.0446% worse rather than at most 1%, only 6
of the required 10 planted motifs were recovered, and active/unique topic counts
fell by 3/4 rather than at most 2/2.

## Decision

Stop. Do not run seeds 23/37, K=128 or real MSnLib. Do not replace the context
scalar with a fixed coefficient or add another mechanism. The result isolates
both contributions: top-2 token restriction repairs much of entmax-only topic
starvation, while leave-one-out spectral context supplies additional recovery
and inventory that the parameter-free route cannot retain. The existing
one-scalar top-2 contextual Routing ETM remains the simplest demonstrated model
that works.

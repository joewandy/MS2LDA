# Literature survey: neural topic models for short, sparse MS/MS spectra

## Scope and question

This survey supports the research handoff on branch `research/etm-ecrtm-msnlib-20260826`. The practical question is deliberately narrower than "invent a better neural topic model":

> Starting from the established MS2LDA representation of tandem mass spectra as bags of fragment and neutral-loss words, what published topic-model family is the most defensible neural successor to LDA, and what is the smallest additional mechanism needed to preserve a broad, chemically useful Mass2Motif inventory on short, sparse spectra?

The literature points to a clean progression: **LDA/MS2LDA -> ETM with pretrained spectral token embeddings -> a published anti-collapse model such as ECRTM if collapse is observed -> only then a targeted MS-specific adaptation**. This is also the progression best supported by the synthetic experiments in this branch.

The central modeling difficulty is unusual. An MS2LDA spectrum is extremely short in terms of distinct observations (often only a few dozen physical peaks / fragment-loss terms), but the current intensity representation converts normalized intensity to repeated pseudo-counts. In the realistic simulator used here, roughly 25-26 nonzero vocabulary terms can correspond to about 1,400 total pseudo-count mass. Thus the model simultaneously sees **short-document co-occurrence sparsity** and a **large reconstruction-weight scale**. A useful model must also discover many distinct topics, because the purpose is Mass2Motif discovery rather than document compression alone.

## 1. Domain foundation: MS2LDA and the spectrum-as-document representation

The original MS2LDA paper introduced the core analogy used throughout this project: tandem mass spectra are documents, fragment and neutral-loss features are words, and recurrent co-occurrence patterns are latent topics called Mass2Motifs. A single spectrum can contain several Mass2Motifs, which is chemically important because one molecule can contain several reusable structural subunits [van der Hooft et al., 2016].

MS2LDA 2.0 retains that formulation but modernizes the implementation and evaluation. It uses Tomotopy's collapsed-Gibbs LDA implementation, and its preprocessing rounds fragment/loss masses to tokens such as `frag@70.04`; normalized peak intensities are discretized to integers from 0 to 100 and represented as repeated words. The 2026 study also introduces Mass2Motif Annotation Guidance (MAG), uses Spec2Vec for downstream structural guidance, and evaluates motif usefulness on MSnLib reference standards [Torres Ortega et al., 2026].

This gives the neural project a strong constraint: **do not solve a different task**. A replacement should still produce an explicit topic-word distribution `beta` interpretable as a Mass2Motif and a per-spectrum topic mixture `theta`, and should be judged by motif inventory and chemical quality, not only held-out likelihood.

MSnLib provides an unusually valuable real-data benchmark because it is a large, open, machine-learning-oriented spectral resource covering 30,008 unique molecules and more than two million MS^n spectra [Brungs et al., 2025]. The current repository's scaffold/compound-disjoint validation and MAG/SOS pipeline should therefore remain the decisive evidence surface.

## 2. Why pretrained spectral token embeddings are a natural ingredient

Spec2Vec is direct evidence that the NLP analogy extends beyond LDA in mass spectrometry. It adapts Word2Vec to fragment and neutral-loss co-occurrence and shows that learned peak/loss embeddings capture structural relationships useful for spectral similarity [Huber et al., 2021].

This makes the repository's train-only SGNS token table scientifically natural rather than an arbitrary engineering feature. It also makes the **pretrained-word-embedding form of ETM** especially attractive: the word embedding is not imported from an unrelated language model, but learned from the same physical co-occurrence domain that defines Mass2Motifs.

The synthetic experiments support this point strongly. Original ETM with jointly learned embeddings collapsed in the overcomplete stress setting, while fixed train-only SGNS embeddings materially stabilized topic usage and improved recovery of the planted spectrum mixtures. That result is consistent with the intended role of pretrained embedding geometry in ETM and with Spec2Vec's demonstration that fragment/loss co-occurrence contains useful chemistry-related structure.

## 3. ETM is the cleanest published neural successor to LDA

The Embedded Topic Model (ETM) explicitly combines topic modeling with word embeddings. Topic-word natural parameters are produced by inner products between topic embeddings and word embeddings, while a neural inference network amortizes document-level inference [Dieng et al., 2020]. This is conceptually close to what MS2LDA needs:

- `beta` remains an explicit distribution over fragment/loss words;
- `theta` is inferred in one neural pass rather than iterative local inference;
- rare and heavy-tailed vocabularies are a stated motivation of ETM;
- pretrained/fixed embeddings are an established ETM configuration, not a new MS-specific invention.

The strongest computational-biology precedent is **scETM**. Zhao et al. explicitly adapt ETM to sparse biological count matrices from single-cell RNA sequencing, with cells playing the role of documents and genes the role of words. scETM retains a neural encoder and interpretable embedding-based decoder, scales beyond one million cells, and reports biologically meaningful topics [Zhao et al., 2021]. The analogy is not exact—gene counts and MS/MS fragment/loss pseudo-counts have different observation processes—but scETM is important reviewer-facing evidence that "ETM + a biological count-data domain" is an accepted model-development strategy.

For a computational-biology paper, this gives a far cleaner starting claim than M1:

> We replace classical LDA inference in MS2LDA with the published Embedded Topic Model and initialize its fragment/loss embedding space from train-only spectral SGNS co-occurrence embeddings.

That statement is easy to locate in an established methodological lineage and makes every later modification attributable.

## 4. Neural variational topic modeling and component collapse

AVITM/ProdLDA established one-pass amortized variational inference as a practical route for topic models and explicitly discussed the component-collapsing problem that can arise when applying autoencoding variational Bayes to topic models [Srivastava and Sutton, 2017]. This is relevant because good reconstruction or perplexity does not guarantee that all topic components remain meaningful.

The synthetic ETM experiment in this branch reproduced exactly that type of failure in a form relevant to Mass2Motif discovery. With 36 learned ETM topics for 18 planted motifs, jointly learned-embedding ETM achieved strong held-out likelihood but materially used only about 8/36 topics and produced highly redundant topic-word distributions. That is **component/topic collapse rather than classic posterior collapse**: the variational document representation remains active, but many topic components become starved or semantically duplicate one another.

This distinction matters for evaluation. For MS2LDA, a model with excellent NLL but only a small effective motif inventory is a scientific failure, because the goal is reusable substructure-pattern discovery. The real campaign should therefore record active-topic counts, corpus/topic entropy, duplicate-topic similarity, and MAG/SOS in addition to NLL.

## 5. Short-document literature explains why MS/MS is a hostile regime

Short-text topic modeling has long recognized that conventional topic models lose information when documents contain too few word co-occurrences. The Biterm Topic Model (BTM) was motivated by this exact problem and models corpus-level word-pair co-occurrence rather than relying only on within-document statistics [Yan et al., 2013]. BTM is not a direct replacement here because MS2LDA needs reliable **per-spectrum multi-topic mixtures**, but its diagnosis is directly relevant.

More recent neural work reaches the same conclusion. Neural Sinkhorn Topic Model (NSTM) states that neural topic models often degrade severely on short documents and uses optimal transport over topic/word embedding geometry to improve both topic quality and document representations [Zhao et al., 2021]. Meta-Complement Topic Model likewise argues that short documents have insufficient word co-occurrence and transfers semantics from longer documents to complement them [Zhang and Lauw, 2022]. Empirical analyses also show that document length materially changes neural-topic-model behavior and that optimizing one metric does not guarantee good performance on another [Terragni and Fersini, 2021].

A recent 2026 short-text model, GLSTM, frames the solution space as a trade-off between global context and local/sharp document-topic structure and explicitly uses sharpening/quantization to obtain more informative short-document topic proportions [Nguyen et al., 2026]. This is conceptually notable because the same tension appeared independently in our experiments: ECRTM repaired the global topic inventory, but its raw spectrum-level `theta` became far too diffuse; simple inference-temperature sharpening repaired much of that mismatch without changing `beta`.

The important conclusion is not that MS/MS should adopt a short-text model wholesale. It is that **our short-spectrum behavior is a known statistical regime**, so failures such as diffuse theta or insufficient co-occurrence need not be justified by a bespoke architecture from first principles.

## 6. Published anti-collapse solutions: what they solve and how closely they fit

### 6.1 ECRTM: the most direct match to the observed failure

ECRTM was proposed specifically because neural topic models can produce semantically collapsed, repetitive topics and an insufficient topic inventory. Its Embedding Clustering Regularization (ECR) encourages each topic embedding to become the center of a separately aggregated word-embedding cluster [Wu et al., 2023].

This maps almost perfectly to our observed learned-ETM failure. In the synthetic overcomplete setting, TopMost-style ECRTM kept 36/36 topics materially occupied, reduced topic redundancy, and improved recovery of the planted motif-word distributions relative to fixed-SGNS ETM. That is why ECRTM should be evaluated as the published anti-collapse candidate rather than transplanting an ECR term into ETM ourselves.

There are two caveats. First, raw ECRTM `theta` was badly over-dispersed on the sparse spectra. Second, ECR uses a full topic-word transport geometry, which may be computationally expensive at real MSnLib scale (`K=1000`, `V~21k`). These are real-data feasibility questions, not reasons to redesign ECRTM before testing it.

### 6.2 Neural Sinkhorn Topic Model: elegant but a larger model-family change

NSTM also uses optimal transport and embedding geometry, but OT is more central to its document/topic inference objective than ECR is in ECRTM [Zhao et al., 2021]. Its explicit attention to short documents makes it scientifically relevant, and it should remain a serious fallback comparator if ETM/ECRTM fail. However, moving directly to NSTM would change more of the core likelihood/inference story than starting from ETM and then testing ECRTM.

For a paper whose main biological contribution is Mass2Motif discovery rather than a new generic topic model, ETM -> ECRTM is therefore the simpler lineage to test first.

### 6.3 Direct Dirichlet/sparse-prior VAEs

Several methods try to represent simplex-valued latent variables more directly. Dirichlet VAE work studies component collapse under Dirichlet latent variables [Joo et al., 2020], while Burkhardt and Kramer explicitly separate sparsity and smoothness in a Dirichlet-VAE topic model [Burkhardt and Kramer, 2019]. These papers make a sparse-prior route defensible if document mixtures remain the primary problem.

Our synthetic experiment, however, showed that merely changing ECRTM's symmetric Dirichlet concentration from 1.0 to 0.1 barely changed spectrum sparsity. The likely reason is specific to the MS2LDA observation representation: intensity pseudo-count reconstruction is orders of magnitude larger than the KL term. This suggests that **prior concentration alone is not the first lever to pull**. A more principled future study would revisit likelihood/count scaling, KL weighting, or the observation model itself rather than repeatedly tuning alpha.

## 7. Co-occurrence regularization and NPMI: useful safeguard, but not the first paper story

Topic coherence has a long literature, and NPMI is widely used because co-occurring top words tend to correspond to more interpretable topics [Lau et al., 2014; Dieng et al., 2020]. The current M1 trainer uses a positive-NPMI graph as an explicit structural regularizer, and the real MSnLib ablations show that removing it **inside M1** materially reduces useful motifs.

That does not establish that every alternative model must include NPMI from the outset. A family-level redesign can behave differently from deleting one mechanism inside a tightly coupled architecture. The first ETM/ECRTM campaign should therefore omit the custom NPMI loss to preserve a recognizable published baseline. If NLL and topic occupancy look acceptable but MAG/SOS falls, **positive co-occurrence regularization becomes the most plausible single targeted mechanism to test next**.

This would be a much stronger experimental argument than carrying NPMI into every candidate preemptively.

## 8. Contextual document representations: a natural later extension, not the first experiment

Contextualized Topic Models show that injecting pretrained contextual document embeddings into neural topic models can improve topic coherence [Bianchi et al., 2021]. In mass spectrometry, DreaMS now provides a direct analogue: a self-supervised Transformer pretrained on millions of tandem mass spectra, yielding reusable spectrum representations that encode molecular structure [Bushuiev et al., 2025].

This creates a clear future path:

> fixed-SGNS ETM/ECRTM -> optionally add a frozen DreaMS spectrum embedding to the document encoder.

That experiment is far easier to motivate than M1's custom token-routing stack because both halves have published precedent: ETM/CTM for topic modeling and DreaMS for MS/MS representation learning. It should nevertheless be deferred until the simple published models are characterized on real MSnLib; otherwise any gain or failure cannot be attributed cleanly.

FASTopic is another relevant modern direction. It reconstructs semantic relations among pretrained document embeddings, topic embeddings, and word embeddings, and regularizes them using an embedding transport plan [Wu et al., 2024]. It is attractive as a later efficiency/stability comparator, but it moves further from the ordinary probabilistic topic-mixture likelihood used by MS2LDA and ETM.

## 9. Where M1 fits in the literature

M1 should not be described as an instance of a single existing architecture. It combines established ideas—embedding geometry, sparse routing, whole-document conditioning, Sinkhorn balancing, co-occurrence regularization, prototype separation, and one-pass inference—in a new MS-specific configuration.

The synthetic studies explain why it remains valuable: M1-like models can produce exceptionally distinct motif-word distributions and resist topic duplication. The committed real MSnLib result also has the strongest chemical evidence available today. It is therefore best treated as:

- the incumbent real-data model;
- an exploratory/upper-bound architecture showing what aggressive Mass2Motif-specific inductive bias can achieve;
- a source of hypotheses about which mechanism to add back if published models fail.

It is **not** the easiest primary model for a computational-biology paper, because each custom mechanism becomes a reviewer-facing design decision that needs separate motivation and ablation.

## 10. How the approaches we explored map onto the literature

| Approach explored | Closest published lineage | Problem addressed | Synthetic finding | Current status |
|---|---|---|---|---|
| Tomotopy LDA / MS2LDA 2.0 | LDA; MS2LDA | Established Mass2Motif discovery | Real incumbent classical baseline | Keep comparator |
| M1 contextual token-routing model | Prototype/routing + OT/co-occurrence ideas; no single exact parent | Short-spectrum context + topic inventory + motif distinctness | Very clean/nonredundant planted motifs; strongest committed real chemistry | Keep incumbent/control, not first paper-facing simplification |
| Pooled/shared deterministic ETM-like simplifications | ETM-inspired embedding decoder, amortized deterministic encoder | Remove bespoke routing/training stack | Some variants improved NLL/theta but blended topics more than M1 | Historical architecture screen |
| Faithful ETM, jointly learned embeddings | ETM | Published neural LDA successor | Reproduced severe component/topic collapse at overcomplete K despite good NLL | Negative/control result |
| Faithful ETM, fixed SGNS | ETM-PWE + Spec2Vec-style spectral embeddings | Stabilize embedding geometry with domain co-occurrence | Strong theta recovery, much less collapse; simplest strong published candidate | **Run first on MSnLib** |
| Direct ETM + ECR transplant | ETM plus manually transplanted ECR | Anti-collapse | Performed worse; 6 active topics in representative stress run | Abandon; poor science story and poor result |
| TopMost-style ECRTM | ECRTM | Topic collapse / duplicate-topic inventory | 36/36 topics active and best planted beta recovery, but diffuse theta | **Run as published anti-collapse candidate** |
| ECRTM alpha=0.1 | Dirichlet/sparse-prior literature | Sparse document mixtures | Little effect because reconstruction scale dominates KL | Negative diagnostic |
| ECRTM inference temperature tau=0.30 | Calibration/sharpening; analogous to short-text quantization intuition | Diffuse per-spectrum theta | Strong improvement in theta sparsity/recovery without changing beta | Predeclared validation-only calibration candidate |
| Neural Sinkhorn Topic Model | NSTM | Short-document representation + coherent/diverse topics | Not yet run in our benchmark | Fallback published family if ETM/ECRTM fail |
| Contextualized topic model + DreaMS | CTM + DreaMS | Missing document-level structural context | Not yet run | Bounded future work after simple models |
| FASTopic | DSR/ETP with pretrained Transformer embeddings | Stability/efficiency/topic discovery | Not yet run | Later comparator, larger departure |

## 11. Recommended research sequence after the literature survey

### First real-data campaign

1. **Fixed-SGNS ETM**, unchanged apart from using the repository's train-only spectral embeddings.
2. **Maintained TopMost ECRTM**, initialized from exactly the same embeddings.
3. Evaluate ECRTM twice: raw `theta`, and the already-frozen synthetic-derived `tau=0.30` inference calibration.
4. Use the exact existing validation split, completion scoring, and leakage-controlled MAG/SOS evaluation.
5. Keep the candidate test split locked until a model passes validation gates.

### If fixed-SGNS ETM passes chemistry gates

Prefer it as the primary scientific model unless ECRTM provides a clearly material chemical improvement. It gives the simplest paper story and strongest connection to ETM/scETM literature.

### If ETM fails mainly through duplicate/starved topics

ECRTM is the directly motivated published replacement. Do not invent an ETM+ECR hybrid; that negative control already failed synthetically.

### If ECRTM topics are good but theta is too diffuse

Report raw and frozen-temperature results. If a real method change is needed beyond calibration, investigate the **observation/KL scaling mismatch** before inventing new routing. Direct Dirichlet/sparse-prior literature provides possible tools, but synthetic alpha tuning alone was insufficient.

### If both published models fit likelihood but lose MAG/SOS chemistry

Test one chemistry/co-occurrence mechanism at a time. Positive-NPMI/co-occurrence structure is the most defensible first add-back because M1's real ablations suggest it has a direct motif-quality role.

### If short-spectrum inference remains the limiting issue

NSTM or a recent short-text model is a more established next family than creating another custom router. If the missing information is instead whole-spectrum structural context, a frozen DreaMS embedding through a conventional contextualized topic-model encoder is the cleaner next experiment.

## 12. Reviewer-facing position

The strongest computational-biology narrative is therefore:

> MS2LDA established LDA as an interpretable model of recurring MS/MS fragment/loss patterns. We investigate whether an established neural topic model can retain that interpretation while amortizing inference and exploiting spectral co-occurrence embeddings. We first adapt ETM using train-only spectral SGNS embeddings, an approach with a direct biological precedent in scETM. Because neural topic models can suffer component/topic collapse—an issue we reproduce under realistic short-sparse MS/MS simulation—we additionally evaluate the published ECRTM anti-collapse model. Any further mass-spectrometry-specific mechanism is introduced only if the locked MSnLib chemical validation demonstrates that it is required.

This makes the novelty primarily **domain adaptation, validation, and biological usefulness**, not an unnecessary claim to have invented a new generic neural topic architecture.

## References

1. van der Hooft JJJ, Wandy J, Barrett MP, Burgess KEV, Rogers S. Topic modeling for untargeted substructure exploration in metabolomics. *PNAS*. 2016. https://pmc.ncbi.nlm.nih.gov/articles/PMC5137707/
2. Torres Ortega LR, Dietrich J, Wandy J, Mol H, van der Hooft JJJ, et al. Large-scale discovery and annotation of substructure patterns in mass spectrometry profiles. *Nature Communications*. 2026;17:8350. doi:10.1038/s41467-026-75038-0.
3. Brungs C, et al. MSnLib: efficient generation of open multi-stage fragmentation mass spectral libraries. *Nature Methods*. 2025. doi:10.1038/s41592-025-02813-0.
4. Huber F, Ridder L, Verhoeven S, Spaaks JH, Diblen F, Rogers S, van der Hooft JJJ. Spec2Vec: Improved mass spectral similarity scoring through learning of structural relationships. *PLOS Computational Biology*. 2021;17(2):e1008724. doi:10.1371/journal.pcbi.1008724.
5. Dieng AB, Ruiz FJR, Blei DM. Topic Modeling in Embedding Spaces. *Transactions of the Association for Computational Linguistics*. 2020;8:439-453. doi:10.1162/tacl_a_00325.
6. Srivastava A, Sutton C. Autoencoding Variational Inference for Topic Models. *ICLR*. 2017. arXiv:1703.01488.
7. Zhao H, Phung D, Huynh V, Le T, Buntine W. Neural Topic Model via Optimal Transport. *ICLR*. 2021. arXiv:2008.13537.
8. Wu X, Dong X, Nguyen TT, Luu AT. Effective Neural Topic Modeling with Embedding Clustering Regularization. *ICML / PMLR*. 2023;202:37335-37357. https://proceedings.mlr.press/v202/wu23c.html
9. Zhao Y, Cai H, Zhang Z, Tang J, Li Y, et al. Learning interpretable cellular and gene signature embeddings from single-cell transcriptomic data. *Nature Communications*. 2021;12:5261. doi:10.1038/s41467-021-25534-2.
10. Yan X, Guo J, Lan Y, Cheng X. A Biterm Topic Model for Short Texts. *WWW*. 2013:1445-1456. doi:10.1145/2488388.2488514.
11. Zhang DC, Lauw HW. Meta-Complementing the Semantics of Short Texts in Neural Topic Models. *NeurIPS*. 2022;35. doi:10.52202/068431-2139.
12. Terragni S, Fersini E. An Empirical Analysis of Topic Models: Uncovering the Relationships between Hyperparameters, Document Length and Performance Measures. *RANLP*. 2021:1408-1416. doi:10.26615/978-954-452-072-4_157.
13. Bianchi F, Terragni S, Hovy D. Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence. *ACL-IJCNLP*. 2021:759-766. doi:10.18653/v1/2021.acl-short.96.
14. Bushuiev R, Bushuiev A, Samusevich R, Brungs C, Sivic J, Pluskal T. Self-supervised learning of molecular representations from millions of tandem mass spectra using DreaMS. *Nature Biotechnology*. 2025. doi:10.1038/s41587-025-02663-3.
15. Wu X, Nguyen T, Zhang DC, Wang WY, Luu AT. FASTopic: Pretrained Transformer is a Fast, Adaptive, Stable, and Transferable Topic Model. 2024. arXiv:2405.17978.
16. Joo W, Lee W, Park S, Moon IC. Dirichlet Variational Autoencoder. *Pattern Recognition*. 2020;107:107514. doi:10.1016/j.patcog.2020.107514.
17. Burkhardt S, Kramer S. Decoupling Sparsity and Smoothness in the Dirichlet Variational Autoencoder Topic Model. *JMLR*. 2019;20(131):1-27.
18. Lau JH, Newman D, Baldwin T. Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality. *EACL*. 2014:530-539. doi:10.3115/v1/E14-1056.
19. Nguyen T, Ngo Van L, Nguyen Duc A, Dinh Viet S. Global and local context in short text neural topic model. *Artificial Intelligence*. 2026;353:104502. doi:10.1016/j.artint.2026.104502.

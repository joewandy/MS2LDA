![header](App/assets/MS2LDA_LOGO_white.jpg)
![Maintainer](https://img.shields.io/badge/maintainer-Rosina_Torres_Ortega-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Jonas_Dietrich-blue)
![Maintainer](https://img.shields.io/badge/maintainer-Joe_Wandy-blue)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.12625409.svg)](https://doi.org/10.5281/zenodo.15858124)

**MS2LDA** is an advanced tool designed for unsupervised substructure discovery in mass spectrometry data, utilizing topic modeling and providing automated annotation of discovered motifs. This tool significantly enhances the capabilities described in the [original MS2LDA paper](https://www.pnas.org/doi/abs/10.1073/pnas.1608041113) (2016), offering users an integrated workflow with improved usability, detailed visualizations, and a searchable motif database (MotifDB).

Mass spectrometry fragmentation patterns hold abundant structural information vital for analytical chemistry, natural product research, and food safety assessments. However, interpreting this data remains challenging, and only a fraction of available information is traditionally utilized. MS2LDA addresses this by identifying recurring substructures (motifs) across spectral datasets without relying on prior compound identification, thus accelerating structure elucidation and analysis.

---

# MS2LDA Installation and Usage

You can install MS2LDA using **pip**, **Conda**, or **Poetry**, depending on your preferences and requirements.

## Quick Install with pip

```bash
pip install ms2lda
```

## Quick Start Demo

Get started with MS2LDA in minutes! See the [**Quick Start Guide**](QUICK_START.md) for step-by-step instructions using Conda, Poetry, or virtualenv.

## Installation Guides

For more detailed installation options and development setup:

- [**Conda Installation Guide**](README_CONDA.md) - Uses Conda environment management.
- [**Poetry Installation Guide**](README_POETRY.md) - Uses Poetry for dependency management (recommended for developers).

---

## Experimental neural MS2LDA reference

The branch contains a deliberately isolated proof-of-concept combining frozen
[DreaMS](https://github.com/pluskal-lab/DreaMS) chemical embeddings with
variational LDA. Tomotopy remains the default production backend; the normal
CLI and dashboard are unchanged.

Create the pinned Python 3.11 environment, then install this checkout without
re-resolving its production dependencies:

```bash
conda env create -f environment-hybrid.yml
conda activate ms2lda-hybrid
python -m pip install --no-deps \
  "git+https://github.com/pluskal-lab/DreaMS.git@dbec3a0b514a99e5056cfccde4559fda8cfe8129"
python -m pip install --no-deps -e .
```

This environment is for the reference model and its tests. Use the normal
MS2LDA installation for the full CLI, dashboard, and annotation workflow.
The explicit `--no-deps` keeps DreaMS's unused full-toolkit packages out of
this environment and prevents its moving `msml@main` dependency from replacing
the exact commit pinned in the YAML file.

The model-specific code is
[`MS2LDA/hybrid_lda.py`](MS2LDA/hybrid_lda.py) and
[`MS2LDA/dreams_features.py`](MS2LDA/dreams_features.py). Here, `documents`
are the existing lists of `frag@...` and `loss@...` MS2LDA words. They must
correspond one-for-one, in the same order, to the matchms `spectra`. The
mathematical specification, limitations, and prespecified validation protocol
are in
[`docs/hybrid_lda_method.tex`](docs/hybrid_lda_method.tex), with a compiled
[`PDF`](output/pdf/hybrid_lda_method.pdf). A minimal usage example follows:

```python
from MS2LDA.dreams_features import DreaMSFeatureExtractor, pool_word_embeddings
from MS2LDA.hybrid_lda import HybridLDAConfig, HybridLDAModel

extractor = DreaMSFeatureExtractor()
features = extractor.extract(spectra)
word_features = pool_word_embeddings(documents, features)

config = HybridLDAConfig(
    num_topics=200,
    embedding_dim=features.spectrum_embeddings.shape[1],
)
model = HybridLDAModel(config)
model.set_word_embeddings(word_features)
for words, embedding in zip(documents, features.spectrum_embeddings, strict=True):
    model.add_doc(words, embedding=embedding)
model.train()
# Optional final frozen-topic phase: train the encoder through two local VB
# updates without changing the learned topics or structured word prior.
model.fit_inference_network()
```

For a new spectrum, extract its DreaMS embedding with the same `extractor`,
then call `make_doc(query_words, embedding=...)` and `infer(..., iter=5)`.
Passing `tolerance=1e-4` makes `iter` a maximum adaptive-refinement budget;
the default `tolerance=None` preserves an exact number of updates.
The model exposes the topic and document accessors needed by Tomotopy-shaped
downstream code. The synthetic benchmark is reproducible through
`scripts/benchmark_semi_amortized_inference.py`; the trusted historical
mushroom-artifact check uses
`scripts/benchmark_mushroom_inference_phase.py`. Their aggregate is in
[`docs/benchmarks/semi_amortized_inference_summary.json`](docs/benchmarks/semi_amortized_inference_summary.json).
Comparative topic discovery and chemical motif quality remain separate
validation questions.

---

## Command Line Tool Usage

MS2LDA provides powerful command-line tools for batch processing and analysis of mass spectrometry data.

For detailed instructions on using the command-line interface, see the [**Command Line Tool Guide**](README_CLI.md).

---

## MS2LDAViz Application

MS2LDA includes a web-based visualization application (MS2LDAViz) for exploring and analyzing results.

For instructions on starting and using the visualization application, see the [**MS2LDAViz Guide**](README_VIZ.md).

---

## MS2LDA Documentation

📚 **[View the full documentation](docs/docs/index.md)**

Our comprehensive documentation includes:
- Getting started guides
- API reference
- Tutorials and examples
- Parameter settings and advanced usage

## Citing MS2LDA

Please cite our work if you use MS2LDA in your research:

Torres Ortega, L.R., Dietrich, J., Wandy, J., Mol, H., & van der Hooft, J.J.J. (2025). Large-scale discovery and annotation of hidden substructure patterns in mass spectrometry profiles. *bioRxiv*. doi: [https://doi.org/10.1101/2025.06.19.659491](https://www.biorxiv.org/content/10.1101/2025.06.19.659491v1)

---

## Our Research Group

[![Github Logo](App/assets/WUR_RGB_standard_2021.png?raw=true)](https://www.wur.nl/en.htm)

---

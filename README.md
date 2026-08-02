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

## Experimental hybrid LDA reference

The isolated [`ms2lda_hybrid`](ms2lda_hybrid) package combines classical
variational LDA topic discovery with frozen DreaMS features and a
semi-amortized new-document inference stage. It does not change the production
`MS2LDA` package, CLI, dashboard, or Tomotopy backend.

The exact model, assumptions, limitations, and validation status are specified
in [`docs/hybrid_lda_method.tex`](docs/hybrid_lda_method.tex). A pinned Python
3.11 environment is provided in [`environment-hybrid.yml`](environment-hybrid.yml).
After creating it, install the DreaMS commit named in the method document and
this checkout with `--no-deps`. The reproducible inference-only experiment is:

```bash
python -m benchmarks.semi_amortized_inference
```

That synthetic experiment tests limited-budget inference under fixed oracle
topics; it is not evidence of improved topic discovery or chemical motif
quality.

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

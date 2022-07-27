[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6914054.svg)](https://doi.org/10.5281/zenodo.6914054)

# CALVADOS
### Coarse-graining Approach to Liquid-liquid phase separation Via an Automated Data-driven Optimisation Scheme 

This repository contains Python code to run coarse-grained molecular dynamics simulations of intrinsically disordered proteins (IDPs) using the CALVADOS model.

### Layout

- `single_chain/` Python code to run single-chain simulations of IDPs using the CALVADOS model. `python submit.py` submits a simulation of a single ACTR chain on a single CPU.
- `direct_coexistence/` Python code to run multi-chain simulations of IDPs using the CALVADOS model in slab geometry. `python submit.py` submits a direct-coexistence simulation of 100 A1 LCD chains on a single GPU.

In the examples, direct-coexistence and single-chain simulations are performed using [openMM](https://openmm.org/) and [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/latest/) installed with [mphowardlab/azplugins](https://github.com/mphowardlab/azplugins), respectively.

### Usage

To run the code, install [Miniconda](https://conda.io/miniconda.html) and make sure all required packages are installed by issuing the following terminal commands

```bash
    conda env create -f environment.yml
    source activate calvados
```

Authors
-------------

[Giulio Tesei (@gitesei)](https://github.com/gitesei)

[Thea K. Schulze (@theaschulze)](https://github.com/theaschulze)

[Ramon Crehuet (@rcrehuet)](https://github.com/rcrehuet)

[Kresten Lindorff-Larsen (@lindorff-larsen)](https://github.com/lindorff-larsen)

Articles
-------------

G. Tesei, T. K. Schulze, R. Crehuet, and K. Lindorff-Larsen. _PNAS_ 118(44), 2021. DOI: [10.1073/pnas.2111696118](https://www.pnas.org/doi/10.1073/pnas.2111696118)

G. Tesei and K. Lindorff-Larsen. _bioRxiv_ 2022. DOI: [10.1101/2022.07.09.499434](https://doi.org/10.1101/2022.07.09.499434)

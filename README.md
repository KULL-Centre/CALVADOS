[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6914054.svg)](https://doi.org/10.5281/zenodo.6914054)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULL-Centre/_2023_Tesei_IDRome/blob/main/IDRLab.ipynb)
[![Video](http://img.shields.io/badge/►-Video-FF0000.svg)](https://youtu.be/r-eFzoBiQZ4)

# CALVADOS
### Coarse-graining Approach to Liquid-liquid phase separation Via an Automated Data-driven Optimisation Scheme 

This repository contains Python code to run coarse-grained molecular dynamics simulations of intrinsically disordered proteins (IDPs) using the CALVADOS model.

### Layout

- `single_chain/` Python code to run single-chain simulations of IDPs using the CALVADOS model. `python submit_local.py` runs a simulation of a single ACTR chain on a single CPU.
- `direct_coexistence/` Python code to run multi-chain simulations of IDPs using the CALVADOS model in slab geometry. `python submit.py` submits a direct-coexistence simulation of 100 A1 LCD chains on a single GPU.

In the examples, direct-coexistence and single-chain simulations are performed using [openMM](https://openmm.org/) and [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/latest/) installed with [mphowardlab/azplugins](https://github.com/mphowardlab/azplugins), respectively.

### Usage

To run the code, install [Miniconda](https://conda.io/miniconda.html) and make sure all required packages are installed by issuing the following terminal commands

```bash
    conda env create -f environment.yml
    source activate calvados
```

#### Commands to install [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/latest/) v2.9.3 with [mphowardlab/azplugins](https://github.com/mphowardlab/azplugins) v0.11.0

```bash
    curl -LO https://github.com/glotzerlab/hoomd-blue/releases/download/v2.9.3/hoomd-v2.9.3.tar.gz
    tar xvfz hoomd-v2.9.3.tar.gz
    git clone https://github.com/mphowardlab/azplugins.git
    cd azplugins
    git checkout tags/v0.11.0
    cd ..
    cd hoomd-v2.9.3
    mkdir build
    cd build
    cmake ../ -DCMAKE_INSTALL_PREFIX=<path to python> \
        -DENABLE_CUDA=ON -DENABLE_MPI=ON -DSINGLE_PRECISION=ON -DENABLE_TBB=OFF \
        -DCMAKE_CXX_COMPILER=<path to g++> -DCMAKE_C_COMPILER=<path to gcc>
    make -j4
    cd ../hoomd
    ln -s ../../azplugins/azplugins azplugins
    cd ../build && make install -j4
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

G. Tesei and K. Lindorff-Larsen. _Open Research Europe_ 2023 2(94). DOI: [10.12688/openreseurope.14967.2](https://doi.org/10.12688/openreseurope.14967.2)

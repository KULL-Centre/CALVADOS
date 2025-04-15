[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6914053.svg)](https://doi.org/10.5281/zenodo.6914053)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULL-Centre/_2023_Tesei_IDRome/blob/main/IDRLab.ipynb)
[![CALVADOS Video](http://img.shields.io/badge/►-Video-FF0000.svg)](https://youtu.be/r-eFzoBiQZ4)
[![IDRome Video](http://img.shields.io/badge/►-Video-FF0000.svg)](https://youtu.be/kL3-cusHgzM)
[![Python application](https://github.com/KULL-Centre/CALVADOS/actions/workflows/python-app.yml/badge.svg)](https://github.com/KULL-Centre/CALVADOS/actions/workflows/python-app.yml)

# CALVADOS

Coarse-grained implicit-solvent simulations of biomolecules in the OpenMM framework.
Earlier implementations of the code are available on [Zenodo](https://zenodo.org/search?q=metadata.subjects.subject%3A%22CALVADOS%22&l=list&p=1&s=10&sort=bestmatch) ([DOI: 10.5281/zenodo.13754000](https://doi.org/10.5281/zenodo.13754000)).

Please cite the following references when using the software:

- G. Tesei, T. K. Schulze, R. Crehuet, K. Lindorff-Larsen. Accurate model of liquid-liquid phase behavior of intrinsically disordered proteins from optimization of single-chain properties. PNAS (2021), 118(44):e2111696118. [DOI: 10.1073/pnas.2111696118](https://doi.org/10.1073/pnas.2111696118)
- G. Tesei, K. Lindorff-Larsen. Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range. _Open Research Europe_ (2022), 2(94). [DOI: 10.12688/openreseurope.14967.2](https://doi.org/10.12688/openreseurope.14967.2)
- F. Cao, S. von Bülow, G. Tesei, K. Lindorff-Larsen. A coarse-grained model for disordered and multi-domain proteins. _Protein Science_ (2024), 33(11):e5172. [DOI: 10.1002/pro.5172](https://doi.org/10.1002/pro.5172)

## Documentation

The software architecture of CALVADOS and illustrative examples are described in:

S. von Bülow*, Y. Yasuda#, F. Cao#, T. K. Schulze#, A. I. Trolle#, A. S. Rauh#, R. Crehuet#, K. Lindorff-Larsen*, G. Tesei* (# equal contribution)
Software package for simulations using the coarse-grained CALVADOS model, arXiv 2025. https://doi.org/10.48550/arXiv.2504.10408

The examples described in the paper can be found in the `examples` folder.

## Installation Instructions

1. Make new conda environment for calvados
``` 
conda create -n calvados python=3.10
conda activate calvados
```
(2. Only needed when planning to use GPUs: Install openmm via conda-force with cudatoolkit. This step can be skipped if running on CPU only.)
```
conda install -c conda-forge openmm=8.2.0 cudatoolkit=11.8
```
3. Clone package and install CALVADOS and its dependencies using pip
``` 
git clone https://github.com/KULL-Centre/CALVADOS.git
cd CALVADOS
pip install .
(or pip install -e .)
```

## Testing

```bash

  python -m pytest
```
The test `test_potentials` simulates two free amino acids, calculates the potential energies based on the saved trajectory and compares these values with those in the OpenMM log file. Other tests check for correct bond order in the RNA model and correct custom restraints.

## Authors

[Sören von Bülow (@sobuelow)](https://github.com/sobuelow)

[Giulio Tesei (@gitesei)](https://github.com/gitesei)

[Fan Cao (@fancaoErik)](https://github.com/fancaoErik)

[Ikki Yasuda (@iyasuda)](https://github.com/iyasuda)

[Arriën Symon Rauh (@ASRauh)](https://github.com/ASRauh)

[Kresten Lindorff-Larsen (@lindorff-larsen)](https://github.com/lindorff-larsen)


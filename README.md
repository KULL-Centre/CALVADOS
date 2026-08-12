# MEM-CALVADOS

This repository contains a branch of the CALVADOS package that implements the MEM-CALVADOS model for flexible proteins at membrane interfaces.

Please cite the following references when using the software:

- R. Saltutti, G. Tesei. MEM-CALVADOS: A residue-level model for flexible proteins at membrane interfaces.
- S. von Bülow, K. Johansson, K. Lindorff-Larsen. AF-CALVADOS: AlphaFold-guided simulations of multi-domain proteins at the proteome level. _Protein Science_ (2026), 35: e70694.
- F. Cao, S. von Bülow, G. Tesei, K. Lindorff-Larsen. A coarse-grained model for disordered and multi-domain proteins. _Protein Science_ (2024), 33(11):e5172. [DOI: 10.1002/pro.5172](https://doi.org/10.1002/pro.5172)
- G. Tesei, K. Lindorff-Larsen. Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range. _Open Research Europe_ (2022), 2(94). [DOI: 10.12688/openreseurope.14967.2](https://doi.org/10.12688/openreseurope.14967.2)
- G. Tesei, T. K. Schulze, R. Crehuet, K. Lindorff-Larsen. Accurate model of liquid-liquid phase behavior of intrinsically disordered proteins from optimization of single-chain properties. PNAS (2021), 118(44):e2111696118. [DOI: 10.1073/pnas.2111696118](https://doi.org/10.1073/pnas.2111696118)

## Documentation

Examples of how to run simulations of single transmembrane proteins can be found in the `examples` folder:

- `examples/single_TMP` contains scripts and input required to run a simulation of GHR where the protein is modeled using CALVADOS 3
- `examples/single_TMP_AF` contains scripts and input required to run a simulation of VDAC1 where the protein is modeled using AF-CALVADOS

The general architecture of the CALVADOS package and other illustrative examples are described in:

S. von Bülow*, Y. Yasuda#, F. Cao#, T. K. Schulze#, A. I. Trolle#, A. S. Rauh#, R. Crehuet#, K. Lindorff-Larsen*, G. Tesei* (# equal contribution)
Software package for simulations using the coarse-grained CALVADOS model, arXiv 2025. https://doi.org/10.48550/arXiv.2504.10408

## Installation Instructions

1. Make new conda environment for calvados
``` 
conda create -n calvados python=3.13
conda activate calvados
```
(2. Only needed when planning to use GPUs: Install openmm via conda-force with cudatoolkit. This step can be skipped if running on CPU only.)
```
conda install -c conda-forge openmm=8.2.0 cudatoolkit=11.8 mdanalysis=2.9 mdtraj=1.11 
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

[Riccardo Saltutti (@riccsalt)](https://github.com/riccsalt)

[Giulio Tesei (@gitesei)](https://github.com/gitesei)

[Sören von Bülow (@sobuelow)](https://github.com/sobuelow)

[Fan Cao (@fancaoErik)](https://github.com/fancaoErik)

[Ikki Yasuda (@iyasuda)](https://github.com/iyasuda)

[Arriën Symon Rauh (@ASRauh)](https://github.com/ASRauh)

[Kresten Lindorff-Larsen (@lindorff-larsen)](https://github.com/lindorff-larsen)


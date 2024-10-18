[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6914053.svg)](https://doi.org/10.5281/zenodo.6914053)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/KULL-Centre/_2023_Tesei_IDRome/blob/main/IDRLab.ipynb)
[![CALVADOS Video](http://img.shields.io/badge/►-Video-FF0000.svg)](https://youtu.be/r-eFzoBiQZ4)
[![IDRome Video](http://img.shields.io/badge/►-Video-FF0000.svg)](https://youtu.be/kL3-cusHgzM)

# CALVADOS

Coarse-grained implicit-solvent simulations of biomolecules in the openMM framework.
Earlier implementations of the code are available on [Zenodo](https://zenodo.org/search?q=metadata.subjects.subject%3A%22CALVADOS%22&l=list&p=1&s=10&sort=bestmatch) ([DOI: 10.5281/zenodo.13754000](https://doi.org/10.5281/zenodo.13754000)).

Please cite the following references when using the software:

- G. Tesei, T. K. Schulze, R. Crehuet, K. Lindorff-Larsen. Accurate model of liquid-liquid phase behavior of intrinsically disordered proteins from optimization of single-chain properties. PNAS (2021), 118(44):e2111696118. [DOI: 10.1073/pnas.2111696118](https://doi.org/10.1073/pnas.2111696118)
- G. Tesei, K. Lindorff-Larsen. Improved predictions of phase behaviour of intrinsically disordered proteins by tuning the interaction range. _Open Research Europe_ (2022), 2(94). [DOI: 10.12688/openreseurope.14967.2](https://doi.org/10.12688/openreseurope.14967.2)
- F. Cao, S. von Bülow, G. Tesei, K. Lindorff-Larsen. A coarse-grained model for disordered and multi-domain proteins. _Protein Science_ (2024), 33(11):e5172. [DOI: 10.1002/pro.5172](https://doi.org/10.1002/pro.5172)

## Installation Instructions

1. Make new conda environment for calvados
``` 
conda create -n calvados python=3.10
conda activate calvados
```
2. Install numba, mdtraj with conda and openmm (they have caused issues with pip install)
```
conda install numba
conda install -c conda-forge mdtraj
conda install -c conda-forge openmm cudatoolkit=11.2
```
3. Clone package and install CALVADOS and its dependencies using pip
``` 
git clone https://github.com/KULL-Centre/CALVADOS.git
cd CALVADOS
pip install .
(or pip install -e .)
```
4. Clean up faulty pip install of scipy:
```
conda install scipy
```

## Contact

Please check out the example folders and the example files `prepare_minimal.py` and `prepare.py`. 
For further questions and inquiries, please contact us.

## Authors

[Sören von Bülow (@sobuelow)](https://github.com/sobuelow)

[Giulio Tesei (@gitesei)](https://github.com/gitesei)

[Fan Cao (@fancaoErik)](https://github.com/fancaoErik)

[Kresten Lindorff-Larsen (@lindorff-larsen)](https://github.com/lindorff-larsen)


# CALVADOS
### Coarse-graining Approach to Liquid-liquid phase separation Via an Automated Data-driven Optimisation Scheme 

This repository contains Python code to run coarse-grained molecular dynamics simulations of intrinsically disordered proteins (IDPs) using the CALVADOS model.

### Layout

- `single_chain/` Python code to run single-chain simulations of IDPs using the CALVADOS model. `python submit.py` submits a simulation of a single ACTR chain on a single CPU.
- `direct_coexistence/` Python code to run multi-chain simulations of IDPs using the CALVADOS model in slab geometry. `python submit.py` submits a direct-coexistence simulation of 100 A1 LCD chains on a single GPU.

In the examples, direct-coexistence and single-chain simulations are performed using openMM and [HOOMD-blue](https://hoomd-blue.readthedocs.io/en/latest/) installed with [mphowardlab/azplugins](https://github.com/mphowardlab/azplugins), respectively.

### Usage

To run the code, install [Miniconda](https://conda.io/miniconda.html) and make sure all required packages are installed by issuing the following terminal commands

```bash
    conda env create -f environment.yml
    source activate calvados
    jupyter-notebook
```

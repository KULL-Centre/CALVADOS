Python code to run direct-coexistence simulations of IDPs using the CALVADOS model.

`python submit_local.py` runs a simulation of 100 hnRNPA1 LCD chains on a single GPU using CUDA.

`python submit_slurm.py` submits a simulation of 100 hnRNPA1 LCD chains on a single GPU using CUDA and the SLURM queue manager.

Please make sure to install openmm v7.5 and have an nvidia GPU with CUDA installed. For installation instructions, please refer to the [openMM documentation](http://docs.openmm.org/latest/userguide/application/01_getting_started.html).

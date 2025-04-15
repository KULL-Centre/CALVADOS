import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

cwd = os.getcwd()
sysname = 'mixed_system'

# set the side length of the slab box
Lx = 15
Lz = 80

# set the saving interval (number of integration steps)
N_save = 100000

# set final number of frames to save
N_frames = 1000

residues_file = f'{cwd}/input/residues_C2RNA.csv'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [Lx, Lx, Lz], # nm
  temp = 293.15, # 20 degrees Celsius
  ionic = 0.15, # molar
  pH = 7.5,
  topol = 'slab',

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing interval, 1 = 10 fs
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CUDA',
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
  slab_eq = True,
  steps_eq = 100*N_save,

  # JOB SETTINGS (ignore if running locally)
  submit = False
)

# PATH
path = f'{cwd}/{sysname}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p {cwd}/data',shell=True)

analyses = f"""
from calvados.analysis import SlabAnalysis
slab_analysis = SlabAnalysis(
    name='mixed_system',
    input_path=f'{cwd}/mixed_system',
    output_path=f'{cwd}/data',
    input_pdb='top.pdb', input_dcd=None,
    centered_dcd='traj.dcd',
    # use proteins as reference for centering
    ref_chains=(0, 199),  # 0-based indexing, inclusive
    ref_name='FUS-RGG3',
    client_chain_list=[(200, 259)],
    client_names=['polyU40'],
    verbose=False
    )
slab_analysis.center(
    start=250,
    center_target='all'
    )
slab_analysis.calc_profiles()
slab_analysis.calc_concentrations()
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  fresidues = residues_file, # residue definitions
  ffasta = f'{cwd}/input/mix.fasta',

  # RNA settings
  rna_kb1 = 1400.0,
  rna_kb2 = 2200.0,
  rna_ka = 4.20,
  rna_pa = 3.14,
  rna_nb_sigma = 0.4,
  rna_nb_scale = 15,
  rna_nb_cutoff = 2.0
)

components.add(name='FUS-RGG3', molecule_type='protein', nmol=200, charge_termini='both')
components.add(name='polyU40', molecule_type='rna', nmol=60)
components.write(path,name='components.yaml')


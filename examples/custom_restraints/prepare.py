import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np

cwd = os.getcwd()
sysname = 'test2'

# set the side length of the cubic box
L = 30

# set the saving interval (number of integration steps)
N_save = 1000

# set final number of frames to save
N_frames = 100

residues_file = f'{cwd}/residues_CALVADOS3.csv'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, L], # nm
  temp = 293, # K
  ionic = 0.15, # molar
  pH = 7.0,
  topol = 'center',

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing interval, 1 = 10 fs
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU', # or CUDA
  restart = None,# 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,

  custom_restraints = True,
  custom_restraint_type = 'harmonic',
  fcustom_restraints = f'{cwd}/cres.txt',
)

# PATH
path = f'{cwd}/{sysname:s}'
subprocess.run(f'mkdir -p {path}',shell=True)

config.write(path,name='config.yaml')

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = True, # apply restraints
  charge_termini = 'both', # charge N or C or both

  # INPUT
  fresidues = residues_file, # residue definitions
  fdomains = f'{cwd}/domains.yaml', # domain definitions (harmonic restraints)
  pdb_folder = f'{cwd}', # directory for pdb and PAE files

  # RESTRAINTS
  restraint_type = 'harmonic', # harmonic or go
  use_com = True, # apply on centers of mass instead of CA
  colabfold = 1, # PAE format (EBI AF=0, Colabfold=1&2)
  k_harmonic = 700., # Restraint force constant
)
components.add(name='4k2u_mod')

components.write(path,name='components.yaml')


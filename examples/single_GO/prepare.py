import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
args = parser.parse_args()

cwd = os.getcwd()
sysname = f'{args.name:s}'

# set the side length of the cubic box
L = 25

# set the saving interval (number of integration steps)
N_save = 100

# set final number of frames to save
N_frames = 1000

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, L], # nm
  temp = 293.15, # 20 degrees Celsius
  ionic = 0.15, # molar
  pH = 7.5, # 7.5
  topol = 'center',

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing interval, 1 = 10 fs
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU', # or CUDA
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
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
  fresidues = f'{cwd}/input/residues.csv', # residue definitions
  pdb_folder = f'{cwd}/input', # directory for pdb and PAE files

  # RESTRAINTS
  restraint_type = 'go', # harmonic or go
  use_com = True, # apply on centers of mass instead of CA
  colabfold = 0, # PAE format (EBI AF=0, Colabfold=1&2)
  k_go = 10., # Restraint force constant
)
components.add(name=args.name)

components.write(path,name='components.yaml')


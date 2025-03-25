import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name_1',nargs='?',required=True,type=str)
parser.add_argument('--name_2',nargs='?',required=True,type=str)
args = parser.parse_args()

cwd = os.getcwd()
sysname = f'{args.name_1:s}_{args.name_2:s}'

# set the side length of the cubic box
L = 30

# set the saving interval (number of integration steps)
N_save = 1000

# set final number of frames to save
N_frames = 1000

residues_file = f'{cwd}/input/residues_CALVADOS3.csv'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, L], # nm
  temp = 293, # K
  ionic = 0.15, # molar
  pH = 7.0, # 7.5
  topol = 'random',

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
subprocess.run(f'mkdir -p data',shell=True)

analyses = f"""
from calvados.analysis import calc_com_traj, calc_contact_map

chainid_dict = dict({args.name_1:s} = 0, {args.name_2:s} = 1)
calc_com_traj(path="{path:s}",sysname="{sysname:s}",output_path="data",residues_file="{residues_file:s}",chainid_dict=chainid_dict,start=100)
calc_contact_map(path="{path:s}",sysname="{sysname:s}",output_path="data",chainid_dict=chainid_dict)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = True, # apply restraints
  charge_termini = 'both', # charge N or C or both

  # INPUT
  fresidues = residues_file, # residue definitions
  ffasta = f'{cwd}/input/idr.fasta', # residue definitions
  fdomains = f'{cwd}/input/domains.yaml', # domain definitions (harmonic restraints)
  pdb_folder = f'{cwd}/input', # directory for pdb and PAE files

  # RESTRAINTS
  restraint_type = 'harmonic', # harmonic or go
  use_com = True, # apply on centers of mass instead of CA
  colabfold = 1, # PAE format (EBI AF=0, Colabfold=1&2)
  k_harmonic = 700., # Restraint force constant
)

components.add(name=args.name_1, restraint=False, use_com=False)
components.add(name=args.name_2)

components.write(path,name='components.yaml')


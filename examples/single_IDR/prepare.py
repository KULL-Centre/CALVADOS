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
L = 30

# set the saving interval (number of integration steps)
N_save = 7000

# set final number of frames to save
N_frames = 1000

residues_file = f'{cwd}/input/residues_CALVADOS2.csv'
fasta_file = f'{cwd}/input/fastalib.fasta'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, L], # nm
  temp = 298, # K
  ionic = 0.15, # molar
  pH = 7.0,
  topol = 'center',

  # INPUT
  fresidues = residues_file, # residue definitions
  ffasta = fasta_file, # domain definitions (harmonic restraints)

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing interval, 1 = 10 fs
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU', # or CUDA
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,

  # JOB SETTINGS (ignore if running locally)
  submit = False
)

# PATH
path = f'{cwd}/{sysname:s}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

analyses = f"""

from calvados.analysis import save_rg

save_rg("{path:s}","{sysname:s}","{residues_file:s}","data",10)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = True, # apply restraints
  charge_termini = 'both', # options: 'N', 'C', 'both' or 'none'
)
components.add(name=sysname, restraint=False)

components.write(path,name='components.yaml')


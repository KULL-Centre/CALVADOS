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
L = 20

# set the saving interval (number of integration steps)
N_save = 1000

# set final number of frames to save
N_frames = 100

residues_file = f'{cwd}/input/residues_CALVADOS2.csv'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, L], # nm
  temp = 293, # K
  ionic = 0.15, # molar
  pH = 7.0, # 7.5

  # MC BAROSTAT SETTINGS
  box_eq = True, # equilibrate box side lengths
  steps_eq = 100*N_save, # number of equilibration steps
  pressure = [0.1,0,0], # pressure along x, y, and z
  boxscaling_xyz = [False,False,True], # whether so scale along x, y, or z

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

analyses = f"""

from calvados.analysis import save_rg

save_rg("{path:s}","{sysname:s}","{residues_file:s}",".",10)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both

  # INPUT
  fresidues = residues_file, # residue definitions
  ffasta = f'{cwd}/input/idr.fasta', # residue definitions
)

components.add(name=args.name)

components.write(path,name='components.yaml')


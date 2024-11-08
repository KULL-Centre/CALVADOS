import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
parser.add_argument('--alpha',nargs='?',required=True,type=float)
args = parser.parse_args()

cwd = os.getcwd()

N_save = 10000
Lx = 30
Ly = Lx
area_per_lipid = .65
N_lipids = int(np.ceil(Lx*Ly/area_per_lipid)*2)
print(N_lipids)

config = Config(
  # GENERAL
  sysname = f'{args.name:s}_{args.alpha:.2f}', # name of simulation system
  box = [Lx, Ly, 120.], # nm
  temp = 310, # 298 K
  ionic = 0.10, # molar
  pH = 7.2, # 7.5
  topol = 'bilayer',

  # RESTRAINTS
  restraint_type = 'harmonic', # harmonic or go
  use_com = True, # apply on centers of mass instead of CA
  colabfold = 1, # PAE format (EBI AF=0, Colabfold=1&2)
  k_harmonic = 700., # Restraint force constant

  # INPUT
  ffasta = f'{cwd}/input/fastalib.fasta', # input fasta file
  fresidues = f'{cwd}/input/residues.csv', # residue definitions
  fdomains = f'{cwd}/input/domains.yaml', # domain definitions (harmonic restraints)
  pdb_folder = f'{cwd}/input', # directory for pdb and PAE files

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing frequency, 1 = 10fs
  steps = 40*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CUDA', # 'CUDA'
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
  bilayer_eq = False,
  zero_lateral_tension = True,
  steps_eq = 100000,
)

# PATH
path = f'{cwd}/{args.name:s}_{args.alpha:.2f}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

saving_interval = N_save*0.01/1e3 # ns

analyses = f"""

from calvados.analysis import calc_bilayer_prop

calc_bilayer_prop(path,"data/{args.name:s}_{args.alpha:.2f}",{saving_interval:g},100,cooke_model=True)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both
)
components.add(name=args.name, molecule_type='cooke_lipid', nmol=N_lipids, alpha=args.alpha*-1, restraint=False)

components.write(path,name='components.yaml')


import os
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
parser.add_argument('--N_prot',nargs='?',required=True,type=int)
args = parser.parse_args()

name = args.name
N_prot = args.N_prot

cwd = os.getcwd()

N_save = 10000
Lx = 30
Ly = Lx
area_per_lipid = .65
N_lipids = int(np.ceil(Lx*Ly/area_per_lipid)*2)
print(N_lipids)

config = Config(
  # GENERAL
  sysname = f'{name}', # name of simulation system
  box = [Lx, Ly, 120.], # nm
  temp = 310, # K
  ionic = 0.10, # molar
  pH = 7.2,
  topol = 'slab',

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing frequency, 1 = 10fs
  steps = 4*N_save, # number of simulation steps
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
path = f'{cwd}/{name}_{N_prot:d}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

saving_interval = N_save*0.01/1e3 # ns

analyses = f"""

from calvados.analysis import calc_cooke_bilayer_prop

calc_bilayer_prop(path,"data/FUS_40",{saving_interval:g},100,cooke_model=True)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both
  # INPUT
  ffasta = f'{cwd}/input/fastalib.fasta', # input fasta file
  fresidues = f'{cwd}/input/residues.csv', # residue definitions
  pdb_folder = f'{cwd}/input', # directory for pdb and PAE files

  # RESTRAINTS
  restraint_type = 'harmonic', # harmonic or go
  use_com = True, # apply on centers of mass instead of CA
  colabfold = 1, # PAE format (EBI AF=0, Colabfold=1&2)
  k_harmonic = 700., # Restraint force constant

)
components.add(name=name, nmol=N_prot, restraint=False)
components.add(name='POPG', molecule_type='cooke_lipid', nmol=N_lipids, alpha=-0.1, restraint=False)

components.write(path,name='components.yaml')


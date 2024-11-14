
import os
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser
from Bio import SeqIO

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
parser.add_argument('--replica',nargs='?',required=True,type=int)
args = parser.parse_args()

cwd = os.getcwd()
N_save = int(5e4)

sysname = f'{args.name:s}_{args.replica:d}'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [20, 20, 270], # nm
  temp = 293,
  ionic = 0.1, # molar
  pH = 7.5,
  topol = 'slab',
  friction_coeff = 0.001,
  slab_width = 40,

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing frequency, 1 = 10fs
  steps = 12000*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CUDA', # 'CUDA'
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
  slab_eq = True,
  steps_eq = 100*N_save,
)

# PATH
path = f'{cwd}/{sysname}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

analyses = f"""
from calvados.analysis import calc_slab_profiles

calc_slab_profiles(path="{path:s}",name="{sysname:s}",output_folder="data",ref_atoms="all",start=0)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = True, # apply restraints
  charge_termini = 'both', # charge N or C or both or none

  # INPUT
  fresidues = f'{cwd}/input/residues_CALVADOS3.csv', # residue definitions
  fdomains = f'{cwd}/input/domains.yaml', # domain definitions (harmonic restraints)
  pdb_folder = f'{cwd}/input', # directory for pdb and PAE files

)

components.add(name=args.name, nmol=100)
components.write(path,name='components.yaml')

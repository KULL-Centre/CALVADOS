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

folder = f'{args.name:s}_{args.replica:d}'

config = Config(
  # GENERAL
  sysname = folder, # name of simulation system
  box = [15, 15, 150.], # nm
  temp = 293,
  ionic = 0.1, # molar
  pH = 7.5,
  topol = 'slab',
  friction_coeff = 0.001,

  # INPUT
  ffasta = f'{cwd}/input/fastalib.fasta', # input fasta file
  fresidues = f'{cwd}/input/residues_CALVADOS2.csv', # residue definitions

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
path = f'{cwd}/{folder}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

analyses = f"""

from calvados.analysis import calc_slab_profiles

calc_slab_profiles(path,"all","all","data/{folder:s}",0)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both or none
)

components.add(name=args.name, nmol=100)

components.write(path,name='components.yaml')


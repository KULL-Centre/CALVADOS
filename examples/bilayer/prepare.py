import os
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser
from Bio import SeqIO

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
parser.add_argument('--gpu_id',nargs='?',required=True,type=int)
args = parser.parse_args()

cwd = os.getcwd()
N_save = int(5e4)
N_frames = 100
Lx = 10
Ly = Lx
area_per_lipid = .6
N_lipids = int(np.ceil(Lx*Ly/area_per_lipid)*2)

sysname = f'{args.name:s}'
residues_file = f'{cwd}/input/residues.csv'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [Lx, Ly, 120.], # nm
  temp = 297.15,
  ionic = 0.15, # molar
  pH = 7,
  topol = 'bilayer',
  bilayer_eq = True,
  friction = 0.01,

  # RUNTIME SETTINGS
  gpu_id = args.gpu_id,
  wfreq = N_save, # dcd writing frequency, 1 = 10fs
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU', # 'CUDA'
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
)

# PATH
path = f'{cwd}/{sysname}'
output_path = f'{path}/data'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p {output_path}',shell=True)

config.write(path,name='config.yaml')

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both

  # INPUT
  ffasta = f'{cwd}/input/fastalib.fasta', # input fasta file
  fresidues = residues_file, # residue definitions
)
components.add(name=args.name, molecule_type='lipid', nmol=N_lipids, restraint=False)
components.write(path,name='components.yaml')


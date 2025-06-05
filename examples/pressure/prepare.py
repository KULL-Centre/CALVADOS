import os
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser
from Bio import SeqIO

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
parser.add_argument('--gpu_id',nargs='?',required=True,type=int)
parser.add_argument('--box_size',nargs='?',required=True,type=float)
parser.add_argument('--replica',nargs='?',required=True,type=int)
args = parser.parse_args()

cwd = os.getcwd()
N_save = int(5e4)
N_frames = 4000
pressurefreq = 20

sysname = f'{args.name:s}_{args.replica:d}'
residues_file = f'{cwd}/input/residues_CALVADOS2.csv'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [args.box_size]*3, # nm
  temp = 293,
  ionic = 0.15, # molar
  pH = 7,
  topol = 'random',
  friction_coeff = 0.01,

  # INPUT
  ffasta = f'{cwd}/input/fastalib.fasta', # input fasta file
  fresidues = f'{cwd}/input/residues_CALVADOS2.csv', # residue definitions

  # RUNTIME SETTINGS
  gpu_id = args.gpu_id,
  wfreq = N_save, # dcd writing frequency, 1 = 10fs
  steps = N_frames*N_save, # number of simulation steps
  pressure_tensor = True, # save Pxx Pyy Pzz
  pressurefreq = pressurefreq,
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CUDA', # 'CUDA'
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True
)

# PATH
path = f'{cwd}/{sysname}'
output_path = f'{path}/data'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p {output_path}',shell=True)

analyses = f"""
from calvados.analysis import calc_com_traj

calc_com_traj(path="{path:s}",sysname="{sysname:s}",output_path="{output_path:s}",residues_file="{residues_file:s}")
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
  fresidues = residues_file, # residue definitions
)

components.add(name=args.name, nmol=100)

components.write(path, name='components.yaml')


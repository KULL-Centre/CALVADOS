import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
parser.add_argument('--pH',nargs='?',required=True,type=float)
args = parser.parse_args()

cwd = os.getcwd()
sysname = f'{args.name:s}'

# set the side length of the cubic box
L = 34.40

# set the saving interval (number of integration steps)
N_save = 7000

# set final number of frames to save
N_frames = 1010

# set solution pH
pH = args.pH

residues_file = f'{cwd}/input/residues_pCALVADOS2.csv'

# set charge on pSer and pThr based on input pH
pKa_dict = dict(SEP=6.01, TPO=6.3)
df_residues = pd.read_csv(residues_file,index_col=0)
for pres in pKa_dict.keys():
    df_residues.loc[pres,'q'] = - 1 - 1 / (1 + 10**(pKa_dict[pres]-pH))
df_residues.to_csv(residues_file)

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [L, L, L], # nm
  temp = 298, # K
  ionic = 0.19, # M
  pH = pH,
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
subprocess.run(f'mkdir -p {cwd}/data',shell=True)

analyses = f"""

from calvados.analysis import save_conf_prop

save_conf_prop(path="{path:s}",name="{sysname:s}",residues_file="{residues_file:s}",output_path="{cwd}/data",start=10,is_idr=True,select='all')
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


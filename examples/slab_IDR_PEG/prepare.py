import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

def get_number_PEG_chains_wv(wv_fraction, mw_peg, volume_simulation_box, NAv=6.022e23):
    """ wv_fraction: weight/volume fraction
    mw_peg: MW of PEG in Da
    volume_simulation_box: box volume in nm3"""
    molarity_peg = wv_fraction*1e4/mw_peg
    return int(np.round(molarity_peg*volume_simulation_box*1e-27*NAv))

  
cwd = os.getcwd()
sysname = 'mixed_system'

# set the side length of the slab box
Lx = 15
Lz = 150

# set the saving interval (number of integration steps)
N_save = 100

# set final number of frames to save
N_frames = 100000


parser = ArgumentParser()
parser.add_argument('--mw',nargs='?',required=True,type=int)
parser.add_argument('--wv',nargs='?',required=True,type=float)
parser.add_argument('--gpu_id',nargs='?',required=True,type=int)
args = parser.parse_args()

N_PEG = get_number_PEG_chains_wv(args.wv,args.mw,Lx*Ly*Lz)
sysname = f'PEG_{args.mw:d}_{args.wv:.2f}'


config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [Lx, Lx, Lz], # nm
  temp = 293.15, # 20 degrees Celsius
  ionic = 0.15, # molar
  pH = 7.5, # 7.5
  topol = 'slab',
  fixed_lambda = 0.2,
  slab_width = 20,
  slab_outer = 25,

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing frequency, 1 = 10fs
  steps = 1*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CUDA', # 'CUDA'
  gpu_id = args.gpu_id, 
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
  slab_eq = False,
  steps_eq = 100000,

  # JOB SETTINGS (ignore if running locally)
  submit = False
)

# PATH
path = f'{cwd}/{sysname:s}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

analyses = f"""

from calvados.analysis import calc_slab_profiles

calc_slab_profiles(path,"{sysname:s}","data","all and not resname PEG",["resname PEG"],100)
"""

config.write(path,name='config.yaml')

components = Components(
  # Defaults
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both
  fresidues = f'{cwd}/residues_C2PEG.csv', # residue definitions
  ffasta = f'{cwd}/peg.fasta'
)
components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both
)
components.add(name='A1', nmol=100, restraint=False)
components.add(name=f'PEG{args.mw:d}', molecule_type='crowder', nmol=N_PEG, restraint=False, charge_termini=False)
components.write(path,name='components.yaml')

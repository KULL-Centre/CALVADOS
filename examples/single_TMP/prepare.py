import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',required=True,type=str)
args = parser.parse_args()

# set index of residue to place in the membrane midplane
ref_bead = 270

# set the side lengths of the cuboidal box
Lx = 30
Ly = 30
Lz = 120

# set the saving interval (number of integration steps)
N_save = 10

# set final number of frames to save
N_frames = 10

cwd = os.getcwd()
sysname = args.name

area_per_lipid = .65
N_lipids = int(np.ceil(Lx*Ly/area_per_lipid)*2)

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [Lx, Ly, 120.], # nm
  temp = 310, # 298 K
  ionic = 0.15, # molar
  pH = 7.5, # 7.5
  topol = 'shift_ref_bead',
  ref_bead = ref_bead,

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
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU', # or CUDA
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = False,
  bilayer_eq = True,
  zero_lateral_tension = True,
  steps_eq = N_save,

  # JOB SETTINGS (ignore if running locally)
  submit = False
)

# PATH
path = f'{cwd}/{sysname:s}'
subprocess.run(f'mkdir -p {path}',shell=True)

config.write(path,name='config.yaml')

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = True, # apply restraints
  charge_termini = 'both', # charge N or C or both
)
components.add(name=args.name)
components.add(name='POPC', molecule_type='lipid', nmol=N_lipids, alpha=-0.3, restraint=False)

components.write(path,name='components.yaml')


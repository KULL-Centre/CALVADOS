import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser

cwd = os.getcwd()
sysname = 'mixed_system'

# set the side length of the slab box
Lx = 20
Lz = 150

# set the saving interval (number of integration steps)
N_save = 100

# set final number of frames to save
N_frames = 100000

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [Lx, Lx, Lz], # nm
  temp = 293.15, # 20 degrees Celsius
  ionic = 0.15, # molar
  pH = 7.5, # 7.5
  topol = 'slab',

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing interval, 1 = 10 fs
  steps = N_frames*N_save, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU',
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,

  # JOB SETTINGS (ignore if running locally)
  submit = False
)

# PATH
path = f'{cwd}/{sysname}'
subprocess.run(f'mkdir -p {path}',shell=True)

config.write(path,name='config.yaml')

components = Components(
  # Defaults
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both
  fresidues = f'{cwd}/residues_C2RNA.csv', # residue definitions
  ffasta = f'{cwd}/mix.fasta',
 
  # RNA settings
  rna_kb1 = 1400.0,
  rna_kb2 = 2200.0,
  rna_ka = 4.20,
  rna_pa = 3.14,
  rna_nb_sigma = 0.4,
  rna_nb_scale = 15,
  rna_nb_cutoff = 2.0
)

components.add(name='polyR30', molecule_type='rna', nmol=25)
components.add(name='FUSRGG3', molecule_type='protein', nmol=100)
components.write(path,name='components.yaml')


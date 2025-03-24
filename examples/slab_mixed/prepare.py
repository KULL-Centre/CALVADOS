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

residues_file = f'{cwd}/residues_C2RNA.csv'

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
  slab_eq = True,
  steps_eq = 100*N_save,

  # JOB SETTINGS (ignore if running locally)
  submit = False
)

# PATH
path = f'{cwd}/{sysname}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

analyses = f"""
from calvados.analysis import SlabAnalysis, calc_com_traj, calc_contact_map

slab = SlabAnalysis(name="{sysname:s}", input_path="{path:s}",
  output_path="data",
  ref_name = "FUSRGG3", ref_chains = (0, 99),
  client_names = ['polyR30'], client_chain_list = [(100,124)],
  verbose=True)

slab.center(start=0, center_target='all') # center_target='ref' for centering only on FUSRGG3
slab.calc_profiles()
slab.calc_concentrations()
print(slab.df_results)
slab.plot_density_profiles()

# heterotypic cmap
chainid_dict = dict(FUSRGG3 = (0,99), polyR30 = (100,124))
calc_com_traj(path="{path:s}",sysname="{sysname:s}",output_path="data",residues_file="{residues_file:s}",chainid_dict=chainid_dict)
calc_contact_map(path="{path:s}",sysname="{sysname:s}",output_path="data",chainid_dict=chainid_dict,is_slab=True)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  restraint = False, # apply restraints
  ext_restraint = False, # apply external restraints
  charge_termini = 'both', # charge N or C or both
  fresidues = residues_file, # residue definitions
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
components.add(name='FUSRGG3', molecule_type='protein', ext_restraint=True, nmol=100)
components.write(path,name='components.yaml')


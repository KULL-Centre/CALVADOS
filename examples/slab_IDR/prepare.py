import os
import calvados as cal
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
residues_file = f'{cwd}/input/residues_CALVADOS2.csv'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [15, 15, 150.], # nm
  temp = 293,
  ionic = 0.1, # molar
  pH = 7.5,
  topol = 'slab',

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

from calvados.analysis import SlabAnalysis, calc_com_traj, calc_contact_map

slab = SlabAnalysis(name="{sysname:s}", input_path="{path:s}",
  output_path="data", ref_name="{sysname:s}", verbose=True)

slab.center(start=0, center_target='all')
slab.calc_profiles()
slab.calc_concentrations()
print(slab.df_results)
slab.plot_density_profiles()
calc_com_traj(path="{path:s}",name="{sysname:s}",output_path="data",residues_file="{residues_file:s}",list_chainids=[np.arange(100)])
calc_contact_map(path="{path:s}",name="{sysname:s}",output_path="data",prot_1_chainids=np.arange(100),prot_2_chainids=np.arange(100),in_slab=True)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  ext_restraint = False, # apply external restraints
  charge_termini = 'both', # charge N or C or both or none

  # INPUT
  ffasta = f'{cwd}/input/fastalib.fasta', # input fasta file
  fresidues = residues_file, # residue definitions
)

components.add(name=args.name, ext_restraint=True, nmol=100)

components.write(path,name='components.yaml')


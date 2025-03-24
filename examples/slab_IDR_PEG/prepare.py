import os
import pandas as pd
from calvados.cfg import Config, Job, Components
import subprocess
import numpy as np
from argparse import ArgumentParser
from scipy.constants import Avogadro

def get_number_PEG_chains_wv(wv_fraction, mw_peg, volume_simulation_box):
    """ wv_fraction: weight/volume fraction in g / 100 mL
    mw_peg: MW of PEG in Da
    volume_simulation_box: box volume in nm3"""
    molarity_peg = wv_fraction*10/mw_peg # M
    volume_simulation_box = volume_simulation_box*1e-24 # L
    return int(np.round(molarity_peg*volume_simulation_box*Avogadro))

cwd = os.getcwd()

# set the side length of the slab box
Lx = 15
Ly = Lx
Lz = 150

# set the saving interval (number of integration steps)
N_save = 200

# set final number of frames to save
N_frames = 1

parser = ArgumentParser()
parser.add_argument('--mw',nargs='?',required=True,type=int)
parser.add_argument('--wv',nargs='?',required=True,type=float)
parser.add_argument('--gpu_id',nargs='?',required=True,type=int)
args = parser.parse_args()

N_PEG = get_number_PEG_chains_wv(args.wv,args.mw,Lx*Ly*Lz)
sysname = f'PEG{args.mw:d}_{args.wv:.0f}'
residues_file = f'{cwd}/input/residues_C2PEG.csv'

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
  steps = N_frames*N_save, # number of simulation steps
  platform = 'CPU', # 'CUDA'
  gpu_id = args.gpu_id,
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
  slab_eq = True,
  steps_eq = 100
)

# PATH
path = f'{cwd}/{sysname:s}'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p data',shell=True)

analyses = f"""
from calvados.analysis import SlabAnalysis, calc_com_traj, calc_contact_map

#slab = SlabAnalysis(name="{sysname:s}", input_path="{path:s}",
#                    output_path="data",
#                    ref_name = "A1", ref_chains = (0, 99),
#                    client_names = ["PEG{args.mw:d}"], client_chain_list = [(100,{99+N_PEG:d})],
#                    verbose=True)

#slab.center(start=0, center_target='ref') # center_target='ref' for centering only on A1
#slab.calc_profiles()
#slab.calc_concentrations()
#print(slab.df_results)
#slab.plot_density_profiles()

# homotypic cmap
chainid_dict = dict(A1 = (0,99))
calc_com_traj(path="{path:s}",sysname="{sysname:s}",output_path="data",residues_file="{residues_file:s}",chainid_dict=chainid_dict)
calc_contact_map(path="{path:s}",sysname="{sysname:s}",output_path="data",chainid_dict=chainid_dict,is_slab=True)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both
  fresidues = residues_file, # residue definitions
  ffasta = f'{cwd}/input/peg.fasta'
)

components.add(name='A1', molecule_type='protein', nmol=100)
components.add(name=f'PEG{args.mw:d}', molecule_type='crowder', nmol=N_PEG, charge_termini=False)
components.write(path,name='components.yaml')

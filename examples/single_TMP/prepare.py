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

ref_bead = 270
tmd_sel = "resid 265 to 287"

cwd = os.getcwd()
N_save = int(5e4)
N_frames = 1000
Lx = 30
Ly = Lx
area_per_lipid = .63
N_lipids = int(np.ceil(Lx*Ly/area_per_lipid)*2)

sysname = f'{args.name:s}_{args.replica:d}'
residues_file = f'{cwd}/input/residues.csv'

config = Config(
  # GENERAL
  sysname = sysname, # name of simulation system
  box = [Lx, Ly, 100.], # nm
  temp = 310,
  ionic = 0.15, # molar
  pH = 7.4,
  topol = 'shift_ref_bead',
  bilayer_eq = True,
  friction = 0.01,
  pressure_coupling = True,
  pressure = [0,0,0],

  # RUNTIME SETTINGS
  wfreq = N_save, # dcd writing frequency, 1 = 10fs
  logfreq = 100*N_save,
  steps = N_frames*N_save, # number of simulation steps
  steps_eq = 30*N_save,
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CUDA', # 'CUDA'
  restart = 'checkpoint',
  frestart = 'restart.chk',
  verbose = True,
)

# PATH
path = f'{cwd}/{sysname}'
output_path = 'data'
subprocess.run(f'mkdir -p {path}',shell=True)
subprocess.run(f'mkdir -p {output_path}',shell=True)
ref_sel="resname TDO or resname TPO"
strip_sel="not (resname SEB or resname CHO or resname PHO or resname MID or resname TDO or resname TPO)"

analyses = f"""
from calvados.analysis import calc_membrane_profiles, calc_domain_angles, calc_domain_rgs, cmap_selection
calc_membrane_profiles("{path}","{sysname}","{output_path}","{residues_file}","{tmd_sel}",0,"{ref_sel}","{strip_sel}")
angle_sels = dict(TMD="{tmd_sel}", D1="resid 52 to 145", D2="resid 146 to 252")
calc_domain_angles("{path}","{sysname}","{output_path}",angle_sels,"{residues_file}",0.5)
rg_sels = dict(ECD="resid 1 to 252", ICD="resid 313 to 639", ICD_GFP="resid 313 to 863", FL="resid 1 to 863")
calc_domain_rgs("{path}","{sysname}","{output_path}","{residues_file}",rg_sels,0.5)
indexrange_sels = dict(protein_pho=("{strip_sel}", "resname PHO"),protein_tail=("{strip_sel}", "{ref_sel}"))
cmap_selection("{path}","{sysname}","{output_path}",indexrange_sels)
"""

config.write(path,name='config.yaml',analyses=analyses)

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  charge_termini = 'None', # charge N or C or both

  # INPUT
  ffasta = f'{cwd}/input/fastalib.fasta', # input fasta file
  fresidues = residues_file, # residue definitions
  fdomains = f'{cwd}/input/domains.yaml', # domain definitions (harmonic restraints)
  pdb_folder = f'{cwd}/input', # directory for pdb and PAE files
)
components.add(name='POPC', molecule_type='lipid', nmol=N_lipids)
components.add(name=args.name, restraint=True, charge_termini='both', ref_bead=ref_bead)
components.write(path,name='components.yaml')


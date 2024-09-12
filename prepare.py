import os
from calvados.cfg import Config, Job, Components
import subprocess

cwd = os.getcwd()

config = Config(
  # GENERAL
  sysname = 'test', # name of simulation system
  box = [25., 25., 150.], # nm
  temp = 293, # K
  ionic = 0.15, # molar
  pH = 7.0,
  topol = 'slab',

  # RESTRAINTS
  restraint_type = 'harmonic', # harmonic or go
  use_com = True, # apply on centers of mass instead of CA
  colabfold = 1, # PAE format (EBI AF=0, Colabfold=1&2)
  k_harmonic = 700., # Restraint force constant

  # INPUT
  ffasta = f'{cwd}/test.fasta', # input fasta file
  fresidues = f'{cwd}/residues_C3.csv', # residue definitions
  fdomains = f'{cwd}/domains.yaml', # domain definitions (harmonic restraints)
  pdb_folder = f'{cwd}/pdbs', # directory for pdb and PAE files

  # RUNTIME SETTINGS
  wfreq = 1000, # dcd writing frequency, 1 = 10fs
  steps = 100000, # number of simulation steps
  runtime = 0, # overwrites 'steps' keyword if > 0
  platform = 'CPU', # 'CUDA'
  restart = None,
  verbose = True,
)

job = Job(
  batch_sys = 'SLURM', # PBS
  envname = 'calvados', # conda environment
  template = 'robust.jinja',
  fbash = '/home/sobuelow/.bashrc', # path to .bashrc
)

# PATH
path = f'{cwd}/{config.config["sysname"]}'
subprocess.run(f'mkdir -p {path}',shell=True)

config.write(path,name='config.yaml')

components = Components(
  # Defaults
  molecule_type = 'protein',
  nmol = 1, # number of molecules
  restraint = False, # apply restraints
  charge_termini = 'both', # charge N or C or both
)
components.add(name='hSUMO_hnRNPA1S', nmol=1, restraint=True)
components.add(name='Gal3', nmol=5, charge_termini='C')

components.write(path,name='components.yaml')

job.write(path,config,components,name='job.sh')
# job.submit(path,njobs=3)

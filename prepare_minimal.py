import os
from calvados.cfg import Config, Job, Components
import subprocess

cwd = os.getcwd()

config = Config(
  # GENERAL
  sysname = 'test_A1', # name of simulation system
  box = [25., 25., 25.], # nm
  temp = 293, # K
  ionic = 0.15, # molar
  pH = 7.0,


  # RUNTIME SETTINGS
  wfreq = 1000, # dcd writing frequency, 1 = 10fs
  steps = 100000, # number of simulation steps
)

# PATH
path = f'{cwd}/{config.config["sysname"]}'
subprocess.run(f'mkdir -p {path}',shell=True)

config.write(path,name='config.yaml')

components = Components(
    # INPUT
  ffasta = f'{cwd}/test.fasta', # input fasta file
  fresidues = f'{cwd}/residues.csv', # residue definitions
)
components.add(name='A1', nmol=1, restraint=False, charge_termini='both')
components.write(path,name='components.yaml')

# navigate to path followed by
# python run.py

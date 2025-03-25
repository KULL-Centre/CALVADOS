from jinja2 import Template

import os
from time import sleep

from importlib import resources

import yaml
import json

###########################

class Config:
    def __init__(self,**params):
        self.params = params
        self.pkg_base = resources.files('calvados')
        self.config = self.load_default_config(self.pkg_base)
        self.load_config()

    def load_config(self):
        for key, val in self.params.items():
            self.config[key] = val

    def write(self,path,name='config.yaml',analyses=''):
        """ Write config file. """
        self.name = name

        with open(f'{path}/{name}','w') as stream:
            yaml.dump(self.config,stream)
        self.write_runfile(path,analyses)

    @staticmethod
    def load_default_config(pkg_base):
        """ Load default config. """
        with open(f'{pkg_base}/data/default_config.yaml','r') as stream:
            default_config = yaml.safe_load(stream)
        # default_config['fresidues'] = f'{pkg_base}/data/residues.csv'
        return default_config

    @staticmethod
    def write_runfile(path,analyses):
        stream = """from calvados import sim
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--path',nargs='?', default='.', const='.', type=str)
    parser.add_argument('--config',nargs='?', default='config.yaml', const='config.yaml', type=str)
    parser.add_argument('--components',nargs='?', default='components.yaml', const='components.yaml', type=str)

    args = parser.parse_args()

    path = args.path
    fconfig = args.config
    fcomponents = args.components

    sim.run(path=path,fconfig=fconfig,fcomponents=fcomponents)
"""
        with open(f'{path}/run.py','w') as f:
            f.write(stream+analyses)

###########################

class Components:
    def __init__(self,**defaults):
        self.pkg_base = resources.files('calvados')
        self.components = {}

        self.components['defaults'] = self.load_default_component(self.pkg_base,defaults)
        self.components['system'] = {}

    def reset_components(self,**kwargs):
        self.components['system'] = {}

    def add(self,**kwargs):
        nm = kwargs['name']
        del kwargs['name']
        self.components['system'][nm] = kwargs

    def write(self,path,name='components.yaml'):
        """ Write component file. """
        self.name = name
        with open(f'{path}/{name}','w') as stream:
            yaml.dump(self.components,stream,sort_keys=False)

    @staticmethod
    def load_default_component(pkg_base,defaults):
        """ Load default config. """
        # package defaults
        with open(f'{pkg_base}/data/default_component.yaml','r') as stream:
            default_component = yaml.safe_load(stream)
        # manual defaults
        for key, val in defaults.items():
            default_component[key] = val
        return default_component

############################

class Job:
    def __init__(self,**kwargs):
        # package defaults
        self.pkg_base = resources.files('calvados')
        with open(f'{self.pkg_base}/data/default_job.yaml','r') as stream:
            self.settings = yaml.safe_load(stream)
        self.settings['folder'] = f'{self.pkg_base}/data/templates'
        for key, val in kwargs.items():
            self.settings[key] = val

    def write(self,path,config,components,name='job.sh'):
        """ Write PBS or SLURM job. """
        self.jobname = name
        file = f'{self.settings["folder"]}/{self.settings["template"]}'
        with open(file,'r') as f:
            submission = Template(f.read())

        with open(f'{path}/{name}', 'w') as submit:
            submit.write(submission.render(
                **self.settings,**config.config,
                fconfig=config.name,
                fcomponents=components.name,
                path=path))

    def submit(self,path,njobs=1):
        if njobs > 1 and self.settings['batch_sys'] == 'PBS':
            raise Exception('Only single jobs supported with PBS.')
        for idx in range(njobs):
            if self.settings['batch_sys'] == 'SLURM':
                os.system(f'sbatch {path}/{self.jobname}')
            elif self.settings['batch_sys'] == 'PBS':
                os.system(f'qsub {path}/{self.jobname}')
            sleep(0.5)

############################

def write_entry(uniprot,entry,pdb_folder):
    with open(f'{pdb_folder}/{uniprot}_info.json','w') as f:
        json.dump(entry,f)

def load_ebi(uniprot,pdb_folder):
    os.system(f'mkdir -p {pdb_folder}')
    with os.popen(f'curl https://alphafold.ebi.ac.uk/api/prediction/{uniprot}') as f:
        entry = f.read()
    entry = json.loads(entry)[0]
    os.system(f'curl -L {entry["pdbUrl"]} -o {pdb_folder}/{uniprot}.pdb')
    os.system(f'curl -L {entry["paeDocUrl"]} -o {pdb_folder}/{uniprot}.json')
    write_entry(uniprot,entry,pdb_folder)

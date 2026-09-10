import json
import os
from importlib import resources
from time import sleep

import yaml
from jinja2 import Template
from pydantic import BaseModel, TypeAdapter

from .inputmodels import ComponentInput, JobInput, SimulationInput


def model_defaults(model: type[BaseModel]) -> dict:
    """Return the defaults declared for non-required model fields."""
    return {
        name: field.get_default(call_default_factory=True)
        for name, field in model.model_fields.items()
        if not field.is_required()
    }


def serialize_fields(model: type[BaseModel], values: dict) -> dict:
    """Validate and serialize selected model fields."""
    serialized = {}
    for name, value in values.items():
        field = model.model_fields[name]
        adapter = TypeAdapter(field.rebuild_annotation())
        validated = adapter.validate_python(value)
        serialized[name] = adapter.dump_python(validated, mode='json')
    return serialized

###########################

class Config:
    def __init__(self,**params):
        validated = SimulationInput.model_validate(params)
        self.config = validated.model_dump(mode='json')

    def write(self,path,name='config.yaml',analyses=''):
        """ Write config file. """
        self.name = name

        with open(f'{path}/{name}','w') as stream:
            yaml.safe_dump(self.config,stream,sort_keys=False)
        self.write_runfile(path,analyses)

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
        self.defaults = {**model_defaults(ComponentInput), **defaults}
        self.components = {
            'defaults': self.defaults,
            'system': {},
        }

    def reset_components(self,**kwargs):
        self.components['system'] = {}

    def add(self,name,**overrides):
        raw = {
            **self.defaults,
            **overrides,
            'name': name,
        }
        validated = ComponentInput.model_validate(raw)
        values = validated.model_dump(mode='json')
        self.components['defaults'] = serialize_fields(
            ComponentInput, self.defaults
        )
        self.components['system'][name] = {
            key: values[key] for key in overrides
        }

    def write(self,path,name='components.yaml'):
        """ Write component file. """
        self.name = name
        with open(f'{path}/{name}','w') as stream:
            yaml.safe_dump(self.components,stream,sort_keys=False)

############################

class Job:
    def __init__(self,**kwargs):
        validated = JobInput.model_validate(kwargs)
        self.settings = validated.model_dump(mode='json')
        self.pkg_base = resources.files('calvados')
        self.settings['folder'] = f'{self.pkg_base}/data/templates'

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

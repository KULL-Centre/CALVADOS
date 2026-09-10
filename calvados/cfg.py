import json
import os
from importlib import resources
from time import sleep
from typing import Any

import yaml
from jinja2 import Template
from pydantic import BaseModel, TypeAdapter

from .inputmodels import ComponentInput, JobInput, SimulationInput, InputPath


def model_defaults(model: type[BaseModel]) -> dict[str, Any]:
    """Return the defaults declared for non-required model fields."""
    return {
        name: field.get_default(call_default_factory=True)
        for name, field in model.model_fields.items()
        if not field.is_required()
    }


def serialize_fields(
        model: type[BaseModel],
        values: dict[str, Any],
) -> dict[str, Any]:
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
    """Build and write a validated simulation configuration.

    Keyword arguments are validated by :class:`SimulationInput` and serialized
    to YAML-compatible values. Writing the configuration also creates the
    companion Python run script.
    """

    def __init__(self, **params: Any) -> None:
        """Validate simulation parameters and store their serialized values."""
        validated = SimulationInput.model_validate(params)
        self.config = validated.model_dump(mode='json')

    def write(
        self,
        path: InputPath,
        name: InputPath = "config.yaml",
        analyses: str = "",
    ) -> None:
        """ Write config file. """
        self.name = name

        with open(f'{path}/{name}','w') as stream:
            yaml.safe_dump(self.config,stream,sort_keys=False)
        self.write_runfile(path,analyses)

    @staticmethod
    def write_runfile(path: InputPath, analyses: str) -> None:
        """Write the simulation runner and append optional analysis code."""
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
    """Build and write validated component definitions.

    Shared defaults are combined with per-component overrides and checked with
    :class:`ComponentInput` before being serialized to ``components.yaml``.
    """

    def __init__(self, **defaults: Any) -> None:
        """Initialize component defaults and an empty system definition."""
        self.defaults = {**model_defaults(ComponentInput), **defaults}
        self.components = {
            'defaults': self.defaults,
            'system': {},
        }

    def reset_components(self, **kwargs: Any) -> None:
        """Remove every component from the current system definition."""
        self.components['system'] = {}

    def add(self,
            name: str,
            **overrides: Any
    ) -> None:
        """Validate and add a named component with optional overrides."""
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

    def write(
            self,
            path: InputPath,
            name: InputPath = "components.yaml"
        ) -> None:
        """ Write component file. """
        self.name = name
        with open(f'{path}/{name}','w') as stream:
            yaml.safe_dump(self.components,stream,sort_keys=False)

############################

class Job:
    """Render and submit batch jobs for a prepared CALVADOS system.

    Job settings select a packaged Jinja template and batch system. The
    rendered script receives paths and values from the associated
    :class:`Config` and :class:`Components` objects.
    """

    def __init__(self, **kwargs: Any) -> None:
        """Validate job settings and locate the packaged templates."""
        validated = JobInput.model_validate(kwargs)
        self.settings = validated.model_dump(mode='json')
        self.pkg_base = resources.files('calvados')
        self.settings['folder'] = f'{self.pkg_base}/data/templates'

    def write(
            self,
            path: InputPath,
            config: Config,
            components: Components,
            name: InputPath = 'job.sh'
        ) -> None:
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

    def submit(self, path: InputPath, njobs: int = 1) -> None:
        """Submit one or more jobs through the configured batch system."""
        if njobs > 1 and self.settings['batch_sys'] == 'PBS':
            raise Exception('Only single jobs supported with PBS.')
        for idx in range(njobs):
            if self.settings['batch_sys'] == 'SLURM':
                os.system(f'sbatch {path}/{self.jobname}')
            elif self.settings['batch_sys'] == 'PBS':
                os.system(f'qsub {path}/{self.jobname}')
            sleep(0.5)

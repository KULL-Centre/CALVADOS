import pytest
import numpy as np
import os
import pandas as pd
from calvados.cfg import Config, Job, Components
from calvados import sim
import subprocess
import numpy as np
import mdtraj as md

import openmm

def bond_check(i: int, j: int):
    """ Define bonded term conditions. """

    condition0 = (i%2 == 0) # phosphate
    condition1 = (j == i+2) # phosphate -- phosphate
    condition2 = (j == i+1) # phosphate -- base

    condition = condition0 and (condition1 or condition2)
    return condition

@pytest.mark.parametrize(
    ("name"),
    [
        ("cres_test"),
    ],
)

def test_cres(name):

    cwd = os.getcwd()

    sysname = f'{name:s}'

    # set the side length of the cubic box
    L = 8

    # set the temperature
    temp = 298

    # set ionic strength
    ionic = 0.15

    # set the saving interval (number of integration steps)
    N_save = 10

    # set final number of frames to save
    N_frames = 10

    residues_file = f'{cwd}/tests/data/residues_CALVADOS2.csv'
    fasta_file = f'{cwd}/tests/data/fastalib.fasta'

    config = Config(
    # GENERAL
    sysname = sysname, # name of simulation system
    box = [L, L, L], # nm
    temp = temp, # K
    ionic = ionic, # molar
    pH = 7.0,
    topol = 'grid',

    # RUNTIME SETTINGS
    wfreq = N_save, # dcd writing interval, 1 = 10 fs
    logfreq = N_save, # log file writing interval, 1 = 10 fs
    steps = N_frames*N_save, # number of simulation steps
    platform = 'CPU', # or CUDA
    restart = None,
    verbose = True,
    report_potential_energy = False, # True,
    random_number_seed = 12345,

    custom_restraints = True,
    custom_restraint_type = 'harmonic',
    fcustom_restraints = f'{cwd}/tests/data/cres.txt',
    )

    # PATH
    path = f'{cwd}/tests/data/{sysname:s}'

    subprocess.run(f'mkdir -p {path}',shell=True)

    config.write(path,name='config.yaml')

    components = Components(
    # Defaults
    nmol = 1, # number of molecules
    fresidues = residues_file, # residue definitions
    ffasta = fasta_file, # domain definitions (harmonic restraints)
    restraint = False,
    charge_termini = 'none',
    )
    components.add(name=name, molecule_type='protein')

    components.write(path,name='components.yaml')

    sim.run(path=path,fconfig='config.yaml',fcomponents='components.yaml')

    system = openmm.XmlSerializer.deserialize(open(f"{cwd}/tests/data/{sysname}/{sysname}.xml").read())

    force = system.getForces()[3]
    N = force.getNumBonds()

    f = force.getBondParameters(0)
    i, j = f[0], f[1]

    assert (N == 1) and (i == 0) and (j == 9)

    # for idx in range(N):
    #     f = force.getBondParameters(idx)
    #     i, j = f[0], f[1]
    #     assert bond_check(i,j)
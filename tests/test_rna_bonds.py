from pathlib import Path

import openmm
import pytest

from calvados import sim
from calvados.cfg import Components, Config

TEST_DATA = Path(__file__).parent / "data"

def bond_check(i: int, j: int):
    """ Define bonded term conditions. """

    condition0 = (i%2 == 0) # phosphate
    condition1 = (j == i+2) # phosphate -- phosphate
    condition2 = (j == i+1) # phosphate -- base

    condition = condition0 and (condition1 or condition2)
    return condition

@pytest.mark.parametrize(
    ("name", "molecule_type"),
    [
        ("rrrrrr", "rna"),
    ],
)

def test_bonds(name, molecule_type, tmp_path: Path):
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

    residues_file = TEST_DATA / "residues_C2RNA.csv"
    fasta_file = TEST_DATA / "fastalib.fasta"

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
    )

    # PATH
    path = tmp_path / sysname
    path.mkdir()

    config.write(path,name='config.yaml')

    components = Components(
    # Defaults
    nmol = 1, # number of molecules
    fresidues = str(residues_file), # residue definitions
    ffasta = str(fasta_file), # domain definitions (harmonic restraints)
    restraint = False,
    charge_termini = 'none',
    )
    components.add(name=name, molecule_type=molecule_type)

    components.write(path,name='components.yaml')

    sim.run(path=path,fconfig='config.yaml',fcomponents='components.yaml')

    system = openmm.XmlSerializer.deserialize(
        (path / f"{sysname}.xml").read_text()
    )

    force = system.getForces()[2]
    N = force.getNumBonds()

    for idx in range(N):
        f = force.getBondParameters(idx)
        i, j = f[0], f[1]
        assert bond_check(i,j)

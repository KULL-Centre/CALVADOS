from pathlib import Path
from types import SimpleNamespace

import numpy as np
import openmm
import pytest

from calvados import sim
from calvados.cfg import Components, Config

TEST_DATA = Path(__file__).parent / "data"


def test_maps_custom_restraints_across_multiple_components() -> None:
    simulation = sim.Sim.__new__(sim.Sim)
    simulation.components = [
        SimpleNamespace(name="A", params=SimpleNamespace(nmol=2), nbeads=3),
        SimpleNamespace(name="B", params=SimpleNamespace(nmol=1), nbeads=4),
        SimpleNamespace(name="C", params=SimpleNamespace(nmol=2), nbeads=5),
    ]
    simulation.config = SimpleNamespace(fcustom_restraints="unused.txt")
    simulation.parse_custom_restraints = lambda _: [
        [["B", 1, 4], ["C", 2, 1], "1.0", "700.0"]
    ]

    simulation.map_custom_restraints()

    assert [component.start_bead for component in simulation.components] == [0, 6, 10]
    assert simulation.custom_restr_abs == [[9, 15, 1.0, 700.0]]


@pytest.mark.parametrize(
    ("endpoint", "message"),
    [
        (["missing", 1, 1], "component 'missing' is not in the system"),
        (["A", 3, 1], "copy 3 is outside the valid range 1-2"),
        (["A", 1, 4], "bead 4 is outside the valid range 1-3"),
        (["A", 0, 1], "copy 0 is outside the valid range 1-2"),
        (["A", 1, 0], "bead 0 is outside the valid range 1-3"),
    ],
)
def test_rejects_invalid_custom_restraint_endpoint(
    endpoint: list,
    message: str,
) -> None:
    simulation = sim.Sim.__new__(sim.Sim)
    simulation.components = [
        SimpleNamespace(name="A", params=SimpleNamespace(nmol=2), nbeads=3)
    ]
    simulation.config = SimpleNamespace(fcustom_restraints="unused.txt")
    simulation.parse_custom_restraints = lambda _: [
        [endpoint, ["A", 1, 1], "1.0", "700.0"]
    ]

    with pytest.raises(ValueError, match=message):
        simulation.map_custom_restraints()


def test_bilayer_placement_stops_after_ntries(monkeypatch: pytest.MonkeyPatch) -> None:
    simulation = sim.Sim.__new__(sim.Sim)
    simulation.bilayergrid = np.zeros((3, 3))
    simulation.box = np.ones(3)
    simulation.pos = []
    simulation.nparticles = 0
    component = SimpleNamespace(name="lipid", xinit=np.zeros((2, 3)))

    monkeypatch.setattr(
        sim.build,
        "build_xybilayer",
        lambda *args, **kwargs: (np.zeros((2, 3)), False),
    )

    with pytest.raises(
        ValueError,
        match="Could not place bilayer component 'lipid' after 2 attempts",
    ):
        simulation.place_bilayer(component, ntries=2)


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

def test_cres(name, tmp_path: Path):
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

    residues_file = TEST_DATA / "residues_CALVADOS2.csv"
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
    frestart = "custom-restart.chk",
    verbose = True,
    report_potential_energy = False, # True,
    random_number_seed = 12345,

    custom_restraints = True,
    custom_restraint_type = 'harmonic',
    fcustom_restraints = str(TEST_DATA / "cres.txt"),
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
    components.add(name=name, molecule_type='protein')

    components.write(path,name='components.yaml')

    sim.run(path=path,fconfig='config.yaml',fcomponents='components.yaml')

    assert (path / "custom-restart.chk").is_file()
    assert not (path / "restart.chk").exists()

    system = openmm.XmlSerializer.deserialize(
        (path / f"{sysname}.xml").read_text()
    )

    force = system.getForces()[3]
    N = force.getNumBonds()

    f = force.getBondParameters(0)
    i, j = f[0], f[1]

    assert (N == 1) and (i == 0) and (j == 9)

    # for idx in range(N):
    #     f = force.getBondParameters(idx)
    #     i, j = f[0], f[1]
    #     assert bond_check(i,j)

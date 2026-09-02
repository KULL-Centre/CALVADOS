from pathlib import Path

import numpy as np
import pytest

from calvados.cfg import Components, Config
from calvados.components import Protein, RNA
from calvados.sim import Sim


TEST_DATA = Path(__file__).parent / "data"


def _make_component(name: str, molecule_type: str) -> Protein | RNA:
    """Create one registered component without building a complete system."""
    components = Components(
        fresidues=str(TEST_DATA / "residues_C2RNA.csv"),
        ffasta=str(TEST_DATA / "fastalib.fasta"),
        restraint=False,
    )
    components.add(name=name, molecule_type=molecule_type)
    config = Config(box=[8, 8, 8], temp=293.15, ionic=0.15, pH=7.0)
    simulation = Sim(".", config.config, components.components)

    simulation.make_components()

    assert len(simulation.components) == 1
    return simulation.components[0]


def test_registers_protein_component() -> None:
    """A protein configuration should create a Protein component."""
    component = _make_component("Y", "protein")

    assert isinstance(component, Protein)
    assert component.name == "Y"
    assert component.nbeads == 1


def test_registers_rna_component() -> None:
    """An RNA configuration should create an RNA component."""
    component = _make_component("rrrrrr", "rna")

    assert isinstance(component, RNA)
    assert component.name == "rrrrrr"
    assert component.nbeads == 12


def test_rna_angle_map_clips_cosine() -> None:
    """Nearly parallel vectors should produce a finite angle."""
    v1 = np.array(
        [-2.2074710981998043e128, 8.27921441558737e127, 1.5416303946906181e128]
    )
    v2 = np.array(
        [-2.207471098202292e128, 8.279214415593619e127, 1.541630394690393e128]
    )
    rna = RNA.__new__(RNA)
    rna.xinit = np.array([v1, np.zeros(3), np.zeros(3), np.zeros(3), v2])

    rna.calc_angmap()

    assert rna.angmap[0] == pytest.approx(0.0)

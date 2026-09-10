from types import SimpleNamespace

import openmm
import pytest

from calvados.sim import Sim, _split_steps


@pytest.mark.parametrize("steps", [1, 7, 10, 11, 105])
def test_split_steps_preserves_requested_total(steps: int) -> None:
    batches = _split_steps(steps)

    assert sum(batches) == steps
    assert len(batches) <= 10
    assert all(batch > 0 for batch in batches)


@pytest.mark.parametrize(
    (
        "box_eq",
        "bilayer_eq",
        "config_box_eq",
        "config_bilayer_eq",
        "barostat_type",
    ),
    [
        (True, False, True, False, openmm.MonteCarloAnisotropicBarostat),
        (False, True, False, True, openmm.MonteCarloMembraneBarostat),
        (False, False, True, False, openmm.MonteCarloAnisotropicBarostat),
        (False, False, False, True, openmm.MonteCarloMembraneBarostat),
    ],
)
def test_adds_requested_barostat(
    box_eq: bool,
    bilayer_eq: bool,
    config_box_eq: bool,
    config_bilayer_eq: bool,
    barostat_type: type,
) -> None:
    simulation = Sim.__new__(Sim)
    simulation.system = openmm.System()
    simulation.yu = openmm.CustomNonbondedForce("0")
    simulation.ah = openmm.CustomNonbondedForce("0")
    simulation.comp_types = []
    simulation.components = []
    simulation.config = SimpleNamespace(
        ext_force=False,
        custom_restraints=False,
        pressure=(1.0, 1.0, 1.0),
        temp=298.0,
        boxscaling_xyz=(True, True, True),
        box_eq=config_box_eq,
        bilayer_eq=config_bilayer_eq,
        pressure_coupling=True,
    )
    simulation.slab_eq = False
    simulation.box_eq = box_eq
    simulation.bilayer_eq = bilayer_eq

    simulation.add_forces_to_system()

    assert any(
        isinstance(force, barostat_type)
        for force in simulation.system.getForces()
    )

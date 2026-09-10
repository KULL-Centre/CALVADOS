import numpy as np
import pytest
from openmm import openmm, unit

from calvados import interactions


def _particle_parameter_names(force: openmm.CustomNonbondedForce) -> list[str]:
    return [
        force.getPerParticleParameterName(i)
        for i in range(force.getNumPerParticleParameters())
    ]


def _bond_parameter_names(force: openmm.CustomBondForce) -> list[str]:
    return [
        force.getPerBondParameterName(i)
        for i in range(force.getNumPerBondParameters())
    ]


def _global_parameters(force: openmm.Force) -> list[tuple[str, float]]:
    return [
        (
            force.getGlobalParameterName(i),
            force.getGlobalParameterDefaultValue(i),
        )
        for i in range(force.getNumGlobalParameters())
    ]


def test_debye_huckel_parameters():
    eps_yu, k_yu = interactions.genParamsDH(temp=298.0, ionic=0.15)

    assert eps_yu == pytest.approx(1.7708537147570556, rel=1e-14)
    assert k_yu == pytest.approx(1.2738143970635982, rel=1e-14)


@pytest.mark.parametrize(
    ("force", "expression", "parameters", "cutoff", "force_group"),
    [
        pytest.param(
            lambda: interactions.init_ah_interactions(1.0, 2.0, 0.2),
            (
                "1.0*select(step(r-2^(1/6)*s),"
                "4*l*((s/r)^12-(s/r)^6-shift),"
                "4*((s/r)^12-(s/r)^6-l*shift)+(1-l)); "
                "l=select(id1+id2,(id1*id2)*0.5*(l1+l2),0.2); "
                "shift=(s/2.0)^12-(s/2.0)^6; s=0.5*(s1+s2)"
            ),
            ["s", "l", "id"],
            2.0,
            0,
            id="ashbaugh-hatch",
        ),
        pytest.param(
            lambda: interactions.init_yu_interactions(1.0, 0.0, 2.0),
            "q*1.0*(exp(-0.0*r)/r-0.5); q=q1*q2",
            ["q"],
            2.0,
            1,
            id="yukawa",
        ),
        pytest.param(
            lambda: interactions.init_cosine_interactions(1.0),
            (
                "prefactor*select(step(r-rc-1.5*s),0,"
                "select(step(r-rc),-1.0*"
                f"(cos({np.pi}*(r-rc)/(2*1.5*s)))^2,-1.0)); "
                "prefactor=select(id1*id2,1-delta(l1*l2),"
                "(id1+id2)*l1*l2); rc=2^(1/6)*s; s=0.5*(s1+s2)"
            ),
            ["s", "l", "id"],
            2**(1/6) + 1.5,
            2,
            id="cosine",
        ),
        pytest.param(
            lambda: interactions.init_charge_nonpolar_interactions(1.0, 2.0),
            (
                "-step(id1+id2)*1.0*alphaq2R3/2*(1/r-1/2.0); "
                "alphaq2R3=alpha1*q2^2*R31+alpha2*q1^2*R32"
            ),
            ["R3", "alpha", "q", "id"],
            2.0,
            1,
            id="charge-nonpolar",
        ),
    ],
)
def test_nonbonded_force_definition(
    force, expression, parameters, cutoff, force_group
):
    force = force()

    assert force.getEnergyFunction() == expression
    assert _particle_parameter_names(force) == parameters
    assert force.getNonbondedMethod() == openmm.CustomNonbondedForce.CutoffPeriodic
    assert force.getCutoffDistance().value_in_unit(unit.nanometer) == pytest.approx(
        cutoff
    )
    assert force.getForceGroup() == force_group
    assert force.usesPeriodicBoundaryConditions()


@pytest.mark.parametrize(
    ("force", "expression", "parameters", "global_parameters"),
    [
        pytest.param(
            lambda: interactions.init_scaled_LJ(1.0, 2.0),
            (
                "select(step(r-2^(1/6)*s),"
                "n*4*eps*l*((s/r)^12-(s/r)^6-shift),"
                "n*4*eps*((s/r)^12-(s/r)^6-l*shift)+n*eps*(1-l)); "
                "shift=(s/rc)^12-(s/rc)^6"
            ),
            ["s", "l", "n"],
            [("eps", 1.0), ("rc", 2.0)],
            id="scaled-lj",
        ),
        pytest.param(
            lambda: interactions.init_scaled_YU(1.0, 0.0, 2.0),
            "n*q*1.0*(exp(-0.0*r)/r-0.5)",
            ["q", "n"],
            [],
            id="scaled-yukawa",
        ),
        pytest.param(
            lambda: interactions.init_wcafene_interactions(1.0),
            (
                "4*1.0*select(step(r-2^(1/6)*s),0,"
                "(s/r)^12-(s/r)^6+1/4)"
                "+ -0.5*kfene*(rinf^2)*log(1-(r/rinf)^2); rinf=1.5*s"
            ),
            ["s", "kfene"],
            [],
            id="wca-fene",
        ),
    ],
)
def test_custom_bond_force_definition(
    force, expression, parameters, global_parameters
):
    force = force()

    assert force.getEnergyFunction() == expression
    assert _bond_parameter_names(force) == parameters
    assert _global_parameters(force) == global_parameters
    assert force.getForceGroup() == 0
    assert force.usesPeriodicBoundaryConditions()


def test_harmonic_force_definitions():
    forces = [
        interactions.init_bonded_interactions(),
        interactions.init_angles(),
    ]

    for force in forces:
        assert force.getForceGroup() == 0
        assert force.usesPeriodicBoundaryConditions()


def test_slab_restraint_definition():
    force = interactions.init_slab_restraints(
        box=[10.0, 20.0, 30.0], k=2.0
    )

    assert force.getEnergyFunction() == (
        "k*abs(periodicdistance(x,y,z,x,y,z0))"
    )
    assert _global_parameters(force) == [("k", 2.0), ("z0", 15.0)]
    assert force.getForceGroup() == 0
    assert force.usesPeriodicBoundaryConditions()


def test_add_harmonic_restraint():
    force = interactions.init_restraints("harmonic")
    force, record = interactions.add_single_restraint(
        force, "harmonic", dij=0.8, k=100.0, i=2, j=5
    )

    i, j, distance, force_constant = force.getBondParameters(0)
    assert force.getNumBonds() == 1
    assert (i, j) == (2, 5)
    assert distance.value_in_unit(unit.nanometer) == pytest.approx(0.8)
    assert force_constant.value_in_unit(
        unit.kilojoules_per_mole / unit.nanometer**2
    ) == pytest.approx(100.0)
    assert force.getForceGroup() == 0
    assert force.usesPeriodicBoundaryConditions()
    assert record == [3, 6, 0.8, 100.0]


def test_add_go_restraint():
    force = interactions.init_restraints("go")
    force, record = interactions.add_single_restraint(
        force, "go", dij=0.8, k=100.0, i=2, j=5
    )

    i, j, parameters = force.getBondParameters(0)
    assert force.getNumBonds() == 1
    assert (i, j) == (2, 5)
    assert parameters == pytest.approx((0.8, 100.0))
    assert force.getEnergyFunction() == (
        "k*(5*(s/r)^12-6*(s/r)^10); s=s; k=k"
    )
    assert _bond_parameter_names(force) == ["s", "k"]
    assert force.getForceGroup() == 0
    assert force.usesPeriodicBoundaryConditions()
    assert record == [3, 6, 0.8, 100.0]

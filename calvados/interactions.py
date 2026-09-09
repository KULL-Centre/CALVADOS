from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from openmm import openmm, unit

RestType = Literal['harmonic', 'go']

GAS_CONSTANT = 8.3145  # J mol^-1 K^-1
ELEMENTARY_CHARGE = 1.6021766  # 10^-19 C
VACUUM_PERMITTIVITY = 8.854188  # 10^-12 F m^-1
AVOGADRO_CONSTANT = 6.02214076  # 10^23 mol^-1


def _calc_relative_permittivity(temp: float) -> float:
    """Calculate the relative permittivity of water at a temperature in K."""
    return (
        5321 / temp
        + 233.76
        - 0.9297 * temp
        + 0.1417 * 1e-2 * temp * temp
        - 0.8292 * 1e-6 * temp**3
    )


def genParamsDH(temp: float, ionic: float) -> tuple[float, float]:
    """Calculate the Yukawa prefactor in kJ/mol and inverse Debye length in nm^-1."""

    if temp <= 0:
        raise ValueError('Temperature [K] must be positive.')
    if ionic < 0:
        raise ValueError('Ionic strength [M] must not be negative.')

    thermal_energy = GAS_CONSTANT*temp*1e-3
    relative_permittivity = _calc_relative_permittivity(temp)
    # The scaled physical constants and factor 1000 give a length in nm.
    bjerrum_length = (
        ELEMENTARY_CHARGE**2
        / (4*np.pi*VACUUM_PERMITTIVITY*relative_permittivity)
        * AVOGADRO_CONSTANT*1000/thermal_energy
    )
    eps_yu = bjerrum_length*thermal_energy
    # AVOGADRO_CONSTANT/10 converts mol L^-1 to particles nm^-3.
    k_yu = np.sqrt(
        8*np.pi*bjerrum_length*ionic*AVOGADRO_CONSTANT/10
    )
    return eps_yu, k_yu

def init_bonded_interactions() -> openmm.HarmonicBondForce:
    """Initialize a periodic harmonic bond force."""

    # harmonic bonds
    hb = openmm.HarmonicBondForce()
    hb.setUsesPeriodicBoundaryConditions(True)

    return hb

def init_ah_interactions(
    eps: float, rc: float, fixed_lambda: float
) -> openmm.CustomNonbondedForce:
    """Initialize the periodic Ashbaugh-Hatch interaction."""

    # intermolecular interactions
    energy_expression = (
        f'{eps}*select(step(r-2^(1/6)*s),'
        '4*l*((s/r)^12-(s/r)^6-shift),'
        '4*((s/r)^12-(s/r)^6-l*shift)+(1-l))'
    )
    parameter_expression = (
        f'; l=select(id1+id2,(id1*id2)*0.5*(l1+l2),{fixed_lambda})'
        f'; shift=(s/{rc})^12-(s/{rc})^6'
        '; s=0.5*(s1+s2)'
    )
    ah = openmm.CustomNonbondedForce(
        energy_expression + parameter_expression
    )

    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')
    ah.addPerParticleParameter('id')

    ah.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setCutoffDistance(rc*unit.nanometer)
    ah.setForceGroup(0)

    print('Ashbaugh-Hatch potential between particles with lambda=1 and sigma=0.68 at',rc*unit.nanometer,end=': ')
    print(4*eps*((0.68/rc)**12-(0.68/rc)**6)*unit.kilojoules_per_mole)
    return ah

def init_yu_interactions(
    eps: float, k: float, rc: float
) -> openmm.CustomNonbondedForce:
    """Initialize the shifted periodic Yukawa interaction."""

    shift = np.exp(-k*rc)/rc
    energy_expression = (
        f'q*{eps}*(exp(-{k}*r)/r-{shift})'
        '; q=q1*q2'
    )
    yu = openmm.CustomNonbondedForce(energy_expression)
    yu.addPerParticleParameter('q')

    print('Debye-Hückel potential between unit charges at',rc*unit.nanometer,end=': ')
    print(eps*shift*unit.kilojoules_per_mole)

    yu.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    yu.setCutoffDistance(rc*unit.nanometer)
    yu.setForceGroup(1)

    return yu

def init_nonbonded_interactions(
    eps_lj: float,
    cutoff_lj: float,
    eps_yu: float,
    k_yu: float,
    cutoff_yu: float,
    fixed_lambda: float,
) -> tuple[openmm.CustomNonbondedForce, openmm.CustomNonbondedForce]:
    """Initialize protein nonbonded forces without restraints."""

    if cutoff_lj <= 0:
        raise ValueError('LJ cutoff must be positive.')
    if cutoff_yu <= 0:
        raise ValueError('YU cutoff must be positive.')

    ah = init_ah_interactions(eps_lj, cutoff_lj, fixed_lambda)
    ah.setName('AH')
    yu = init_yu_interactions(eps_yu, k_yu, cutoff_yu)
    yu.setName('YU')

    return ah, yu

def init_angles() -> openmm.HarmonicAngleForce:
    """Initialize a periodic harmonic angle force."""

    ha = openmm.HarmonicAngleForce()
    ha.setUsesPeriodicBoundaryConditions(True)
    return ha

def init_lipid_interactions(
    eps_lj: float, eps_yu: float, cutoff_yu: float, factor: float = 1.9
) -> tuple[openmm.CustomNonbondedForce, openmm.CustomNonbondedForce]:
    """Initialize cosine and charge-nonpolar lipid interactions."""

    # harmonic angles
    cos = init_cosine_interactions(factor*eps_lj)
    cn = init_charge_nonpolar_interactions(eps_yu, cutoff_yu)
    return cos, cn

def init_wcafene(eps_lj: float) -> openmm.CustomBondForce:
    """Initialize WCA-FENE interactions with an energy scale of 3*eps_lj."""

    return init_wcafene_interactions(3*eps_lj)

def init_restraints(
    restraint_type: RestType,
) -> openmm.HarmonicBondForce | openmm.CustomBondForce:
    """Initialize a periodic harmonic or Go restraint force."""

    if restraint_type == 'harmonic':
        cs = openmm.HarmonicBondForce()
    elif restraint_type == 'go':
        go_expr = 'k*(5*(s/r)^12-6*(s/r)^10)'
        cs = openmm.CustomBondForce(go_expr+'; s=s; k=k')#; shift=(0.5*(s)/rc)^12-(0.5*(s)/rc)^6')
        cs.addPerBondParameter('s')
        cs.addPerBondParameter('k')
    else:
        raise ValueError("restraint_type must be harmonic or go.")
    cs.setUsesPeriodicBoundaryConditions(True)
    return cs

def init_scaled_LJ(eps_lj: float, cutoff_lj: float) -> openmm.CustomBondForce:
    """Initialize scaled Ashbaugh-Hatch bonded interactions."""

    energy_expression = (
        'select(step(r-2^(1/6)*s),'
        'n*4*eps*l*((s/r)^12-(s/r)^6-shift),'
        'n*4*eps*((s/r)^12-(s/r)^6-l*shift)+n*eps*(1-l))'
        '; shift=(s/rc)^12-(s/rc)^6'
    )
    scLJ = openmm.CustomBondForce(energy_expression)
    scLJ.addGlobalParameter('eps',eps_lj*unit.kilojoules_per_mole)
    scLJ.addGlobalParameter('rc',float(cutoff_lj)*unit.nanometer)
    scLJ.addPerBondParameter('s')
    scLJ.addPerBondParameter('l')
    scLJ.addPerBondParameter('n')
    scLJ.setUsesPeriodicBoundaryConditions(True)
    return scLJ

def init_scaled_YU(
    eps_yu: float, k_yu: float, cutoff_yu: float
) -> openmm.CustomBondForce:
    """Initialize scaled Yukawa bonded interactions."""

    shift = np.exp(-k_yu*cutoff_yu)/cutoff_yu
    energy_expression = (
        f'n*q*{eps_yu}*'
        f'(exp(-{k_yu}*r)/r-{shift})'
    )
    scYU = openmm.CustomBondForce(energy_expression)
    scYU.addPerBondParameter('q')
    scYU.addPerBondParameter('n')
    scYU.setUsesPeriodicBoundaryConditions(True)
    return scYU

def init_slab_restraints(
        box: NDArray[np.float64],
        k: float,
        axis: Sequence[bool] = (False, False, True),
) -> openmm.CustomExternalForce:
    """Initialize restraints toward the box center along selected axes."""

    if len(box) != 3:
        raise ValueError("box argument must have length 3.")
    if len(axis) != 3:
        raise ValueError("axis argument must have length 3.")

    x = 'x0' if axis[0] else 'x'
    y = 'y0' if axis[1] else 'y'
    z = 'z0' if axis[2] else 'z'

    rcent_expr = f'k*abs(periodicdistance(x,y,z,{x},{y},{z}))'
    rcent = openmm.CustomExternalForce(rcent_expr)
    rcent.addGlobalParameter('k',k*unit.kilojoules_per_mole/unit.nanometer)

    for idx, a0 in enumerate([x,y,z]):
        if axis[idx]:
            rcent.addGlobalParameter(a0,box[idx]/2.*unit.nanometer) # center of box in axis dim.
    return rcent

def add_single_restraint(
        cs: openmm.HarmonicBondForce | openmm.CustomBondForce,
        restraint_type: RestType,
        dij: float, k: float,
        i: int, j: int,
) -> tuple[openmm.HarmonicBondForce | openmm.CustomBondForce, list[int | float]]:
    """Add one harmonic or Go restraint and return its one-based record."""

    if restraint_type == 'harmonic':
        cs.addBond(
                i,j, dij*unit.nanometer,
                k*unit.kilojoules_per_mole/(unit.nanometer**2))
    elif restraint_type == 'go':
        cs.addBond(
                i,j, [dij*unit.nanometer,
                k*unit.kilojoules_per_mole])
    else:
        raise ValueError("restraint_type must be harmonic or go")
    restr_pair = [i+1, j+1, dij, k] # 1-based
    return cs, restr_pair

def add_scaled_lj(
    scLJ: openmm.CustomBondForce, i: int, j: int, offset: int, comp
) -> tuple[openmm.CustomBondForce, list[int | float]]:
    """Add one scaled Ashbaugh-Hatch bond and its one-based record."""

    s = 0.5 * (comp.sigmas[i] + comp.sigmas[j])
    l = 0.5 * (comp.lambdas[i] + comp.lambdas[j])
    scLJ.addBond(i+offset,j+offset, [s*unit.nanometer, l*unit.dimensionless, comp.bondscale[i,j]*unit.dimensionless])
    scaled_pair = [i+offset+1, j+offset+1, s, l, comp.bondscale[i,j]] # 1-based
    return scLJ, scaled_pair

def add_scaled_yu(
    scYU: openmm.CustomBondForce, i: int, j: int, offset: int, comp
) -> tuple[openmm.CustomBondForce, list[int | float]]:
    """Add one scaled Yukawa bond and its one-based record."""

    qij = comp.qs[i] * comp.qs[j] * unit.dimensionless
    scYU.addBond(i+offset, j+offset, [qij, comp.bondscale[i,j]*unit.dimensionless])
    scaled_pair = [i+offset+1, j+offset+1, comp.bondscale[i,j]] # 1-based
    return scYU, scaled_pair

def add_exclusion(
    force: openmm.CustomNonbondedForce, i: int, j: int
) -> openmm.CustomNonbondedForce:
    """Add a particle-pair exclusion to a nonbonded force."""
    force.addExclusion(i,j)
    return force

def init_wcafene_interactions(eps: float) -> openmm.CustomBondForce:
    """Initialize the periodic WCA-FENE bond interaction."""

    wca_expression = (
        f'4*{eps}*select(step(r-2^(1/6)*s),0,'
        '(s/r)^12-(s/r)^6+1/4)'
    )
    fene_expression = (
        '+ -0.5*kfene*(rinf^2)*log(1-(r/rinf)^2)'
        '; rinf=1.5*s'
    )
    wcafene = openmm.CustomBondForce(wca_expression+fene_expression)
    wcafene.addPerBondParameter('s')
    wcafene.addPerBondParameter('kfene')
    wcafene.setUsesPeriodicBoundaryConditions(True)
    return wcafene

def init_cosine_interactions(eps: float) -> openmm.CustomNonbondedForce:
    """Initialize the Cooke-Deserno cosine interaction."""

    cosine_expression = (
        'prefactor*select(step(r-rc-1.5*s),0,'
        f'select(step(r-rc),-{eps}*'
        f'(cos({np.pi}*(r-rc)/(2*1.5*s)))^2,-{eps}))'
    )
    parameter_expression = (
        '; prefactor=select(id1*id2,1-delta(l1*l2),'
        '(id1+id2)*l1*l2)'
        '; rc=2^(1/6)*s'
        '; s=0.5*(s1+s2)'
    )
    cosine = openmm.CustomNonbondedForce(
        cosine_expression + parameter_expression
    )
    cosine.addPerParticleParameter('s')
    cosine.addPerParticleParameter('l')
    cosine.addPerParticleParameter('id')
    cosine.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cosine.setCutoffDistance((2**(1/6)+1.5)*unit.nanometer)
    cosine.setForceGroup(2)
    return cosine

def init_charge_nonpolar_interactions(
    eps: float, rc: float
) -> openmm.CustomNonbondedForce:
    """Initialize the lipid charge-nonpolar interaction."""

    energy_expression = (
        f'-step(id1+id2)*{eps}*alphaq2R3/2*(1/r-1/{rc})'
        '; alphaq2R3=alpha1*q2^2*R31+alpha2*q1^2*R32'
    )
    cn = openmm.CustomNonbondedForce(energy_expression)
    cn.addPerParticleParameter('R3')
    cn.addPerParticleParameter('alpha')
    cn.addPerParticleParameter('q')
    cn.addPerParticleParameter('id')
    cn.setNonbondedMethod(openmm.CustomNonbondedForce.CutoffPeriodic)
    cn.setCutoffDistance(rc*unit.nanometer)
    cn.setForceGroup(1)
    return cn

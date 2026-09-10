import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, TypeAlias

import mdtraj as md
import numpy as np
import openmm
from numpy.typing import NDArray
from openmm import app, unit
from tqdm import tqdm
from yaml import safe_load

from . import build, interactions
from .components import (
    COMPONENT_REGISTRY,
    Component,
)
from .inputmodels import InputPath, SimulationInput, validate_inputs

FloatArray: TypeAlias = NDArray[np.float64]

def _split_steps(steps: int, max_batches: int = 10) -> list[int]:
    """Split simulation steps into at most ``max_batches`` nonempty batches."""
    nbatches = min(max_batches, steps)
    batch, remainder = divmod(steps, nbatches)
    return [batch + (index < remainder) for index in range(nbatches)]


class Sim:
    """Build and run a coarse-grained CALVADOS simulation.

    The class validates configuration and component inputs, constructs the
    OpenMM system and MDTraj topology, places every molecule, and applies the
    requested bonded, nonbonded, restraint, equilibration, and reporting
    settings. :meth:`build_system` prepares files and forces, and
    :meth:`simulate` executes or resumes the trajectory.
    """

    def __init__(
            self,
            path: InputPath,
            config: Mapping[str, Any],
            components: Mapping[str, Any],
        ) -> None:
        """Validate inputs and initialize mutable simulation state."""

        self.path = Path(path)
        self.config: SimulationInput
        self.config, self.comp_dict = validate_inputs(config, components)

        # Config options that can change within sim are stored as copied to attributes
        self.box: FloatArray = np.array(self.config.box, dtype=float)
        self.eps_lj: float = float(self.config.eps_lj) * 4.184 # kcal to kJ/mol
        self.slab_eq = self.config.slab_eq
        self.bilayer_eq = self.config.bilayer_eq
        self.box_eq = self.config.box_eq
        self.restart_path = Path(self.config.frestart)

        if not self.restart_path.is_absolute():
            self.restart_path = self.path / self.restart_path

        if self.config.restart in ["pdb", "cif"] and not self.restart_path.is_file():
            raise FileNotFoundError(
                f"{self.config.restart} restart file not found: {self.restart_path}"
            )

        if self.config.restart == 'checkpoint' and self.restart_path.is_file():
            self.slab_eq = False
            self.bilayer_eq = False
            self.box_eq = False

        if self.slab_eq:
            self.rcent = interactions.init_slab_restraints(
                self.box,
                self.config.k_eq,
                self.config.slab_eq_axis
            )

        if self.config.ext_force:
            self.rcent = openmm.CustomExternalForce(self.config.ext_force_expr)

    def make_components(self) -> None:
        """Instantiate components and initialize their properties and restraints."""
        self.components: list[Component] = [] # np.empty(0)
        self.use_restraints = False

        for name, comp_params in self.comp_dict.items():
            component_class = COMPONENT_REGISTRY[comp_params.molecule_type]
            comp = component_class(name, comp_params)

            comp.calc_properties(
                pH=self.config.pH,
                verbose=self.config.verbose,
                eps_lj = self.eps_lj
            )
            if comp.params.restraint:
                self.use_restraints = True
                comp.init_restraint_force() # type: ignore
                if comp.params.restraint_type == 'go':
                    comp.init_scaled_nonbonded( # type: ignore
                        cutoff_lj=self.config.cutoff_lj,
                        cutoff_yu=self.config.cutoff_yu,
                        eps_yu=self.eps_yu,
                        k_yu = self.k_yu,
                    )
            self.components.append(comp)
            # self.components = np.append(self.components, comp)

    def count_components(self) -> None:
        """ Count components and molecules. """

        self.ncomponents: int = 0
        self.nmolecules: int = 0
        self.nmols_per_comp_type: dict[str,int] = {}
        self.comp_types: set[str] = set()

        for comp in self.components:
            self.ncomponents += 1
            self.nmolecules += comp.params.nmol

        print(f'Total number of components in the system: {self.ncomponents}')
        print(f'Total number of molecules in the system: {self.nmolecules}')

        if (
            ((self.ncomponents > 1) or (self.nmolecules > 1))
                and (self.config.topol in ['single', 'center'])
        ):
            raise ValueError("Topol 'single/center' incompatible with multiple molecules.")

        for comp in self.components:
            self.nmols_per_comp_type[comp.params.molecule_type] = self.nmols_per_comp_type.get(comp.params.molecule_type, 0) + comp.params.nmol
            if comp.params.nmol > 0:
                self.comp_types.add(comp.params.molecule_type)

    def reorder_components(self) -> None:
        """Place solute component types before lipids and crowders."""
        self.solute_types = ["protein", "rna", "cyclic", "seastar", "ptm_protein"]
        self.nsolutes = sum(
            self.nmols_per_comp_type.get(comp_type, 0)
            for comp_type in self.solute_types
        )

        # Re-order "solutes" before non-"solutes"
        solute_components: list[Component] = []
        non_solute_components: list[Component] = []

        for comp in self.components:
            if comp.params.molecule_type in self.solute_types:
                solute_components.append(comp)
            else:
                non_solute_components.append(comp)

        self.components = solute_components + non_solute_components

    def build_system(self) -> None:
        """
        Set up system
        * component definitions
        * build particle coordinates
        * define interactions
        * set restraints
        """

        self.top = md.Topology()
        self.system = openmm.System()
        a, b, c = build.build_box(self.box[0],self.box[1],self.box[2])
        self.system.setDefaultPeriodicBoxVectors(a, b, c)


        # init interaction parameters (required before make components)
        self.eps_yu, self.k_yu = interactions.genParamsDH(self.config.temp,self.config.ionic)

        # make components
        self.make_components()
        self.count_components()

        # init interactions
        self.ah, self.yu = interactions.init_nonbonded_interactions(
            self.eps_lj,
            self.config.cutoff_lj,
            self.eps_yu,
            self.k_yu,
            self.config.cutoff_yu,
            self.config.fixed_lambda,
        )
        if "lipid" in self.comp_types:
            self.cos, self.cn = interactions.init_lipid_interactions(
            self.eps_lj,self.eps_yu,self.config.cutoff_yu,factor=1.9
            )
        if "cooke_lipid" in self.comp_types:
            if "lipid" in self.comp_types:
                raise ValueError(
                    "lipid and cooke_lipid components cannot both be present"
                )
            self.cos, self.cn = interactions.init_lipid_interactions(
            self.eps_lj,self.eps_yu,self.config.cutoff_yu,factor=3.0
            )

        self.nparticles = 0 # bead counter
        self.grid_counter = 0 # molecule counter for xy and xyz grids

        self.pos: list[float] = []


        if self.config.topol == 'slab': # proteins + rna
            assert self.config.slab_width is not None
            slab_box = np.array(
                [self.box[0], self.box[1], self.config.slab_width], dtype=float
            )
            self.xyzgrid = build.build_xyzgrid(self.nsolutes, slab_box)
            self.xyzgrid += np.asarray([0,0,self.box[2]/2.-self.config.slab_width/2.])
            if "crowder" in self.comp_types: # crowder
                crowder_box = np.array(
                    [self.box[0], self.box[1], self.box[2]/2.-self.config.slab_outer],
                    dtype=float,
                )
                xyzgrid = build.build_xyzgrid(
                    np.ceil(self.nmols_per_comp_type["crowder"]/2.),
                    crowder_box,
                )
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid, axis=0)
                self.xyzgrid = np.append(
                    self.xyzgrid,
                    xyzgrid + np.asarray([0,0,self.box[2]/2.+self.config.slab_outer]),
                    axis=0
                )
        elif self.config.topol == 'grid':
            self.xyzgrid = build.build_xyzgrid(self.nmolecules,self.box)
        if "lipid" in self.comp_types:
            self.bilayergrid = build.build_xygrid(
                int(self.nmols_per_comp_type["lipid"]*1.05),
                self.box
            )
            if self.nsolutes > 0:
                outer_box = np.array(
                    [self.box[0], self.box[1], self.box[2]/2.-self.box[0]],
                    dtype=float,
                )
                xyzgrid = build.build_xyzgrid(np.ceil(self.nsolutes/2.), outer_box)
                self.xyzgrid = np.append(
                    xyzgrid,
                    xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]),
                    axis=0
                )
        if "cooke_lipid" in self.comp_types:
            self.bilayergrid = build.build_xygrid(
                int(self.nmols_per_comp_type["cooke_lipid"]*1.05),
                self.box,
            )
            if self.nsolutes > 0:
                outer_box = np.array(
                    [self.box[0], self.box[1], self.box[2]/2.-self.box[0]],
                    dtype=float,
                )
                xyzgrid = build.build_xyzgrid(np.ceil(self.nsolutes/2.), outer_box)
                self.xyzgrid = np.append(
                    xyzgrid,
                    xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]),
                    axis=0
                )

        for cidx, comp in enumerate(self.components):
            for idx in range(comp.params.nmol):
                if self.config.verbose:
                    print(f'Component {cidx}, Molecule {idx}: {comp.name}')
                # particle definitions
                self.add_mdtraj_topol(comp)
                self.add_particles_system(comp.mws)

                # add interactions + restraints
                if comp.params.molecule_type in [
                    'protein',
                    'crowder',
                    'cyclic',
                    'seastar',
                    'ptm_protein',
                    'rna',
                ]:
                    _ = self.place_molecule(comp)
                elif comp.params.molecule_type in ['lipid','cooke_lipid']:
                    _ = self.place_bilayer(comp)
                self.add_interactions(comp)

                # add restraints towards box center
                if (self.slab_eq or self.config.ext_force) and comp.params.ext_restraint:
                    self.add_ext_restraints(comp)

        if self.config.custom_restraints:
            self.map_custom_restraints()
            self.add_custom_restraints()

        trajectory = md.Trajectory(self.pos, self.top, 0, self.box, [90,90,90])

        self.pdb_cg = f'{self.path}/top.pdb'
        self.cif_cg = f'{self.path}/top.cif'
        if self.config.restart not in ["pdb", "cif"]: # restart checkpoint or None
            trajectory.save_pdb(self.pdb_cg)
            trajectory.save_cif(self.cif_cg)

        self.add_forces_to_system()
        self.print_system_summary()

    def add_forces_to_system(self) -> None:
        """ Add forces to system. """

        # Intermolecular forces
        for force in [self.yu, self.ah]:
            self.system.addForce(force)

        if ("lipid" in self.comp_types) or ("cooke_lipid" in self.comp_types):
            for force in [self.cos, self.cn]:
                self.system.addForce(force)

        # Intramolecular forces
        for comp in self.components:
            comp.get_forces() # bonded, angles, restraints...
            for force in comp.forces:
                self.system.addForce(force)
            if comp.params.restraint:
                print(f'Number of restraints for comp {comp.name}: {comp.cs.getNumBonds()}') # type: ignore

        # External force
        if self.config.ext_force:
            self.system.addForce(self.rcent)

        # Equilibration forces
        if self.slab_eq:
            self.system.addForce(self.rcent)

        # Custom forces
        if self.config.custom_restraints:
            self.system.addForce(self.cres)
            print(f'Number of custom restraints: {self.cres.getNumBonds()}')

        # barostat force, for equilibration and/or production
        if self.box_eq or (self.config.box_eq and self.config.pressure_coupling):
            assert self.config.pressure is not None
            barostat = openmm.MonteCarloAnisotropicBarostat(
                [
                    self.config.pressure[0] * unit.bar,
                    self.config.pressure[1] * unit.bar,
                    self.config.pressure[2] * unit.bar
                ],
                self.config.temp * unit.kelvin,
                self.config.boxscaling_xyz[0],
                self.config.boxscaling_xyz[1],
                self.config.boxscaling_xyz[2],
                1000,
            )
            self.system.addForce(barostat)

        # Bilayer eq. force, for equilibration and/or production
        if self.bilayer_eq or (self.config.bilayer_eq and self.config.pressure_coupling):
            assert self.config.pressure is not None
            barostat = openmm.MonteCarloMembraneBarostat(
                self.config.pressure[0] * unit.bar,
                0.0 * unit.bar * unit.nanometer,
                self.config.temp * unit.kelvin,
                openmm.MonteCarloMembraneBarostat.XYIsotropic,
                openmm.MonteCarloMembraneBarostat.ZFixed,
                10000,
            )
            self.system.addForce(barostat)

    def print_system_summary(self, write_xml: bool = True) -> None:
        """ Print system information and write xml. """

        if write_xml:
            with open(f'{self.path}/{self.config.sysname}.xml', 'w') as output:
                output.write(openmm.XmlSerializer.serialize(self.system))

        print(f'{self.nparticles} particles in the system')
        print('---------- FORCES ----------')
        print(f'ah: {self.ah.getNumParticles()} particles, {self.ah.getNumExclusions()} exclusions')
        print(f'yu: {self.yu.getNumParticles()} particles, {self.yu.getNumExclusions()} exclusions')
        if self.slab_eq:
            print(f'Equilibration restraints (rcent) towards box center in {self.config.slab_eq_axis} direction')
            print(f'rcent: {self.rcent.getNumParticles()} restraints')
        if self.bilayer_eq:
            print('Equilibration under zero lateral tension')
        if self.box_eq:
            axes = [
                axis
                for axis, enabled in zip("XYZ", self.config.boxscaling_xyz)
                if enabled
            ]
            print(
                "Equilibration through changes in box side lengths along "
                + " and ".join(axes)
            )
    def place_molecule(
            self,
            comp: Component,
            ntries: int = 10000
        ) -> FloatArray:
        """
        Place proteins based on topology.
        """

        if self.config.topol == 'slab':
            x0 = self.xyzgrid[self.grid_counter]
            # x0[2] = self.box[2] / 2. # center in z
            xs = x0 + comp.xinit
            self.grid_counter += 1
        elif self.config.topol == 'grid':
            x0 = self.xyzgrid[self.grid_counter]
            xs = x0 + comp.xinit
            self.grid_counter += 1
        elif self.config.topol == 'center':
            x0 = self.box * 0.5 # place in center of box
            xs = x0 + comp.xinit
        elif self.config.topol == 'shift_ref_bead':
            x0 = self.box * 0.5 # place in center of box
            xs = x0 + comp.xinit
            xs -= comp.xinit[self.config.ref_bead]
        else:
            xs_others = np.array(self.pos, dtype=float)
            xs = build.random_placement(self.box, xs_others, comp.xinit, ntries=ntries)
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return np.array(xs, dtype=np.float64)

    def place_bilayer(self, comp: Component, ntries: int = 10000) -> FloatArray:
        """
        Place proteins based on topology.
        """

        inserted = False
        attempts = 0
        xs: FloatArray | None = None
        while not inserted and attempts < ntries and self.bilayergrid.size > 0:
            attempts += 1
            xs_others = np.array(self.pos, dtype=np.float64)
            xs, inserted = build.build_xybilayer(self.bilayergrid[0], self.box, xs_others, comp.xinit)
            if not inserted:
                xs, inserted = build.build_xybilayer(self.bilayergrid[0], self.box, xs_others, comp.xinit, upward=False)
            if not inserted:
                idx = np.random.randint(self.bilayergrid.shape[0])
                self.bilayergrid[0] = self.bilayergrid[idx]
                self.bilayergrid = np.delete(self.bilayergrid,idx,axis=0)
        if not inserted:
            raise ValueError(
                f"Could not place bilayer component {comp.name!r} after "
                f"{attempts} attempts."
            )

        assert xs is not None
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return xs # positions of the comp (to be used for restraints)

    def add_bonds(self, comp: Component, offset: int) -> None:
        """ Add bond forces. """

        exclusion_map = comp.add_bonds(offset)
        self.add_exclusions(exclusion_map)

    def add_angles(self, comp: Component, offset: int) -> None:
        """ Add bond forces. """

        exclusion_map = comp.add_angles(offset) # type: ignore
        self.add_exclusions(exclusion_map)

    def add_restraints(
            self,
            comp: Component,
            offset: int,
            exclude_nonbonded: bool = True,
        ) -> None:
        """ Add restraints to single molecule. """

        exclusion_map = comp.add_restraints(offset) # type: ignore
        if exclude_nonbonded: # exclude ah, yu when restraining
            self.add_exclusions(exclusion_map)

    def add_custom_restraints(self, exclude_nonbonded: bool = True) -> None:
        """Add configured custom restraints and optional nonbonded exclusions."""
        exclusion_map = []
        # self.custom_restr_pairs = []
        self.cres = interactions.init_restraints(self.config.custom_restraint_type)
        for i, j, r, k in self.custom_restr_abs: # i, j, r, k
            self.cres, restr_pair = interactions.add_single_restraint(
                self.cres, self.config.custom_restraint_type, r, k, i, j)
            # self.custom_restr_pairs.append(restr_pair)
            exclusion_map.append([i,j])
        if exclude_nonbonded: # exclude cres when restraining
            self.add_exclusions(exclusion_map)

    def add_exclusions(self, exclusion_map: list[Any]) -> None:
        """Apply particle-pair exclusions to every relevant nonbonded force."""
        # exclude LJ, YU for restrained pairs
        for excl in exclusion_map:
            self.ah = interactions.add_exclusion(self.ah, excl[0], excl[1])
            self.yu = interactions.add_exclusion(self.yu, excl[0], excl[1])
            if ("lipid" in self.comp_types) or ("cooke_lipid" in self.comp_types):
                self.cos.addExclusion(excl[0], excl[1])
                self.cn.addExclusion(excl[0], excl[1])

    def add_interactions(self, comp: Component) -> None:
        """
        Protein interactions for one molecule of composition comp
        """

        # Get indices of current comp in context of system
        offset = self.nparticles - comp.nbeads 

        # Add Ashbaugh-Hatch
        for sig, lam in zip(comp.sigmas, comp.lambdas):
            if comp.params.molecule_type in ['lipid', 'cooke_lipid']:
                self.ah.addParticle([sig*unit.nanometer, lam, 0])
            elif comp.params.molecule_type == 'crowder':
                self.ah.addParticle([sig*unit.nanometer, lam, -1])
            else: # protein, RNA
                self.ah.addParticle([sig*unit.nanometer, lam, 1])
            if ("lipid" in self.comp_types) or ("cooke_lipid" in self.comp_types):
                if comp.params.molecule_type in ['lipid', 'cooke_lipid']:
                    self.cos.addParticle([sig*unit.nanometer, lam, 0])
                else:
                    self.cos.addParticle([sig*unit.nanometer, lam, 1])
        # Add Debye-Huckel
        for q in comp.qs:
            self.yu.addParticle([q])

        # Add Charge-Nonpolar Interaction
        if ("lipid" in self.comp_types) or ("cooke_lipid" in self.comp_types):
            id_cn = 1 if comp.params.molecule_type == 'protein' else -1
            for sig, alpha, q in zip(comp.sigmas, comp.alphas, comp.qs):
                self.cn.addParticle([(sig/2)**3, alpha, q, id_cn])

        # Add bonds
        self.add_bonds(comp, offset)

        if comp.params.molecule_type == 'rna':
            self.add_angles(comp, offset)

        # Add restraints
        if comp.params.restraint:
            self.add_restraints(comp,offset)

        # write lists
        if self.config.verbose:
            comp.write_bonds(self.path)
            if comp.params.restraint:
                comp.write_restraints(self.path) # type: ignore

    def add_ext_restraints(self, comp: Component) -> None:
        """ Add external-potential restraints. """

        offset = self.nparticles - comp.nbeads # to get indices of current comp in context of system
        for i in range(0,comp.nbeads):
            self.rcent.addParticle(i+offset)

    def add_mdtraj_topol(self, comp: Component) -> None:
        """ Add one molecule to mdtraj topology. """

        # Note: Move this to component eventually.
        chain = self.top.add_chain()

        if comp.params.molecule_type == 'rna':
            for idx, resname in enumerate(comp.seq):
                res = self.top.add_residue(resname, chain, resSeq=idx+1)
                self.top.add_atom(resname+"P", element=md.element.phosphorus, residue=res)
                self.top.add_atom(resname+"N", element=md.element.nitrogen, residue=res)
            for i in range(comp.nbeads-1):
                for j in range(1,comp.nbeads):
                    if comp.bond_check(i,j):
                        self.top.add_bond(chain.atom(i), chain.atom(j))
        else:
            for idx, resname in enumerate(comp.seq):
                if comp.params.molecule_type in ['protein','crowder']:
                    resname = str(comp.residues.loc[resname,'three'])
                res = self.top.add_residue(resname, chain, resSeq=idx+1)
                self.top.add_atom('CA', element=md.element.carbon, residue=res)
            for i in range(chain.n_atoms-1):
                for j in range(i+1, chain.n_atoms):
                    if comp.bond_check(i,j):
                        self.top.add_bond(chain.atom(i), chain.atom(j))

    def add_particles_system(self, mws: FloatArray) -> None:
        """ Add particles of one molecule to openMM system. """

        for mw in mws:
            self.system.addParticle(mw*unit.amu)

    def map_custom_restraints(self) -> None:
        """ Map input format for custom restraints to absolute bead number """
        custom_restr = self.parse_custom_restraints(self.config.fcustom_restraints)
        total_beads = 0
        for comp in self.components:
            comp.start_bead = int(total_beads)
            total_beads += comp.params.nmol * comp.nbeads
        self.custom_restr_abs = []
        components_by_name = {comp.name: comp for comp in self.components}
        for i,j,r,k in custom_restr:
            print(i,j,r,k)
            crestr = []
            # convert beads i, j to absolute bead ids in simulation
            for x in [i,j]:
                name, copy, bead = x[0], x[1], x[2] # 1-based
                if name not in components_by_name:
                    raise ValueError(
                        f"Custom restraint component {name!r} is not in the system."
                    )
                comp = components_by_name[name]
                if not 1 <= copy <= comp.params.nmol:
                    raise ValueError(
                        f"Custom restraint copy {copy} is outside the valid range "
                        f"1-{comp.params.nmol} for component {name!r}."
                    )
                if not 1 <= bead <= comp.nbeads:
                    raise ValueError(
                        f"Custom restraint bead {bead} is outside the valid range "
                        f"1-{comp.nbeads} for component {name!r}."
                    )
                assert comp.start_bead is not None
                x_abs = comp.start_bead + (copy-1)*comp.nbeads + (bead-1)
                crestr.append(x_abs)
            crestr.append(float(r))
            crestr.append(float(k))
            self.custom_restr_abs.append(crestr)

    @staticmethod
    def parse_custom_restraints(fcustom_restraints: InputPath) -> list[Any]:
        """Parse custom-restraint records while preserving one-based indices."""
        custom_restraints: list[Any] = []
        with open(fcustom_restraints,'r') as f:
            for line in f.readlines():
                spl = line.split('|')
                i = spl[0].split()
                j = spl[1].split()
                r = spl[2].split()[0]
                k = spl[2].split()[1]
                restr = [
                    [i[0], int(i[1]), int(i[2])],
                    [j[0], int(j[1]), int(j[2])],
                    r,
                    k
                ]
                custom_restraints.append(restr) # 1-based
        return custom_restraints

    def simulate(self) -> None:
        """ Simulate. """

        append = False

        if self.config.restart == "pdb":
            pdb = app.PDBFile(str(self.restart_path))
        elif self.config.restart == "cif":
            pdb = app.PDBxFile(str(self.restart_path))
        else:
            pdb = app.PDBxFile(self.cif_cg)

        # use langevin integrator
        integrator = openmm.LangevinMiddleIntegrator(
            self.config.temp * unit.kelvin,
            self.config.friction_coeff / unit.picosecond,
            0.01*unit.picosecond
        )
        if self.config.random_number_seed is not None:
            integrator.setRandomNumberSeed(self.config.random_number_seed)
        print(integrator.getFriction(),integrator.getTemperature())

        # assemble simulation
        platform = openmm.Platform.getPlatformByName(self.config.platform)
        if self.config.platform == 'CPU':
            simulation = app.simulation.Simulation(
                pdb.topology,
                self.system,
                integrator,
                platform,
                dict(Threads=str(self.config.threads))
            )
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
                platform.setPropertyDefaultValue('DeviceIndex',str(self.config.gpu_id))
            simulation = app.simulation.Simulation(
                pdb.topology,
                self.system,
                integrator,
                platform
            )
        print('Running on', platform.getName())

        if (self.restart_path.is_file()) and (self.config.restart == 'checkpoint'):
            if not os.path.isfile(f'{self.path}/{self.config.sysname:s}.dcd'):
                raise FileNotFoundError(
                    f'Did not find {self.path}/{self.config.sysname:s}.dcd trajectory to append to!'
                )
            append = True
            print(f'Reading checkpoint file {self.restart_path}')
            print(f'Appending trajectory to {self.path}/{self.config.sysname:s}.dcd')
            print(f'Appending log file to {self.path}/{self.config.sysname:s}.log')
            simulation.loadCheckpoint(str(self.restart_path))
        else:
            if self.config.restart in ['pdb','cif']:
                print(f'Reading in system configuration {self.restart_path}')
            elif self.config.restart == 'checkpoint':
                print(f'No checkpoint file {self.restart_path} found: Starting from new system configuration')
            elif self.config.restart is None:
                print('Starting from new system configuration')

            if os.path.isfile(f'{self.path}/{self.config.sysname:s}.dcd'): # backup old dcd if not restarting from checkpoint
                now = datetime.now()
                dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")
                print(f'Backing up existing {self.path}/{self.config.sysname:s}.dcd to {self.path}/backup_{self.config.sysname:s}_{dt_string}.dcd')
                os.system(f'mv {self.path}/{self.config.sysname:s}.dcd {self.path}/backup_{self.config.sysname:s}_{dt_string}.dcd')
            print(f'Writing trajectory to new file {self.path}/{self.config.sysname:s}.dcd')
            simulation.context.setPositions(pdb.positions)
            print('Minimizing energy.')
            simulation.minimizeEnergy()

        if self.slab_eq:
            print(f"Starting slab equilibration with k_eq == {self.config.k_eq:.4f} kJ/(mol*nm) for {self.config.steps_eq} steps", flush=True)
            simulation.reporters.append(
                app.dcdreporter.DCDReporter(
                    f'{self.path}/equilibration_{self.config.sysname:s}.dcd',
                    self.config.wfreq,
                    append=append,
                )
            )
            simulation.step(self.config.steps_eq)
            state_final = simulation.context.getState(getPositions=True)
            with open(f'{self.path}/equilibration_final.pdb', 'w') as f:
                app.PDBFile.writeFile(simulation.topology, state_final.getPositions(), f)
            with open(f'{self.path}/equilibration_final.cif', 'w') as f:
                app.PDBxFile.writeFile(simulation.topology, state_final.getPositions(), f)
            pdb = app.pdbxfile.PDBxFile(f'{self.path}/equilibration_final.cif')

            for index, force in enumerate(self.system.getForces()):
                if isinstance(force, openmm.CustomExternalForce):
                    print(f'Removing external force {index}')
                    self.system.removeForce(index)
                    break
            integrator = openmm.LangevinIntegrator(
                self.config.temp * unit.kelvin,
                self.config.friction_coeff / unit.picosecond,
                0.01 * unit.picosecond,
            )
            if self.config.random_number_seed is not None:
                integrator.setRandomNumberSeed(self.config.random_number_seed)
            if self.config.platform == 'CPU':
                simulation = app.simulation.Simulation(
                    pdb.topology,
                    self.system,
                    integrator,
                    platform,
                    dict(Threads=str(self.config.threads))
                )
            else:
                simulation = app.simulation.Simulation(
                    pdb.topology,
                    self.system,
                    integrator,
                    platform
                )
            simulation.context.setPositions(pdb.positions)
            print(f'Minimizing energy.')
            simulation.minimizeEnergy()

        if self.box_eq or self.bilayer_eq:
            print(f"Starting pressure equilibration for {self.config.steps_eq} steps", flush=True)
            simulation.reporters.append(
                app.dcdreporter.DCDReporter(
                    f'{self.path}/equilibration_{self.config.sysname:s}.dcd',
                    self.config.wfreq,
                    append=append,
                )
            )
            simulation.step(self.config.steps_eq)
            state_final = simulation.context.getState(getPositions=True,enforcePeriodicBox=True)
            with open(f'{self.path}/equilibration_final.pdb', 'w') as f:
                app.PDBFile.writeFile(simulation.topology, state_final.getPositions(), f)
            with open(f'{self.path}/equilibration_final.cif', 'w') as f:
                app.PDBxFile.writeFile(simulation.topology, state_final.getPositions(), f)
            pdb = app.pdbxfile.PDBxFile(f'{self.path}/equilibration_final.cif')
            topology = pdb.getTopology()
            a, b, c = state_final.getPeriodicBoxVectors()
            topology.setPeriodicBoxVectors(state_final.getPeriodicBoxVectors())
            for index, force in enumerate(self.system.getForces()):
                print(index,force)
            if not self.config.pressure_coupling:
                for index, force in enumerate(self.system.getForces()):
                    if isinstance(force, openmm.MonteCarloMembraneBarostat):
                        print(f'Removing barostat {index}')
                        self.system.removeForce(index)
                        break
                    if isinstance(force, openmm.MonteCarloAnisotropicBarostat):
                        print(f'Removing barostat {index}')
                        self.system.removeForce(index)
                        break
            for index, force in enumerate(self.system.getForces()):
                print(index,force)
            integrator = openmm.LangevinIntegrator(
                self.config.temp * unit.kelvin,
                self.config.friction_coeff / unit.picosecond,
                0.01 * unit.picosecond
            )
            if self.config.random_number_seed is not None:
                integrator.setRandomNumberSeed(self.config.random_number_seed)
            if self.config.platform == 'CPU':
                simulation = app.simulation.Simulation(
                    topology,
                    self.system,
                    integrator,
                    platform,
                    dict(Threads=str(self.config.threads)),
                )
            else:
                simulation = app.simulation.Simulation(
                    topology,
                    self.system,
                    integrator,
                    platform,
                )
            simulation.context.setPositions(state_final.getPositions())
            simulation.context.setPeriodicBoxVectors(a, b, c)

        # run simulation
        simulation.reporters.append(
            app.dcdreporter.DCDReporter(
                f'{self.path}/{self.config.sysname:s}.dcd',
                self.config.wfreq,
                append=append,
            )
        )
        simulation.reporters.append(
            app.statedatareporter.StateDataReporter(
                f'{self.path}/{self.config.sysname}.log',
                self.config.logfreq,
                step=True,
                speed=True,
                elapsedTime=True,
                potentialEnergy=self.config.report_potential_energy,
                separator='\t',
                append=append,
            )
        )

        print("STARTING SIMULATION", flush=True)
        if self.config.runtime is not None: # in unit.hours
            simulation.runForClockTime(
                self.config.runtime*unit.hour,
                checkpointFile=str(self.restart_path),
                checkpointInterval=30*unit.minute,
            )
        else:
            assert self.config.steps is not None
            for batch in tqdm(_split_steps(self.config.steps), mininterval=1):
                simulation.step(batch)
                simulation.saveCheckpoint(str(self.restart_path))

        simulation.saveCheckpoint(str(self.restart_path))

        now = datetime.now()
        dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")

        state_final = simulation.context.getState(
            getPositions=True,
            enforcePeriodicBox=True,
        )
        with open(f'{self.path}/checkpoint.pdb', 'w') as f:
            app.PDBFile.writeFile(simulation.topology, state_final.getPositions(), f)
        with open(f'{self.path}/checkpoint.cif', 'w') as f:
            app.PDBxFile.writeFile(simulation.topology, state_final.getPositions(), f)

def run(
        path: InputPath = '.',
        fconfig: InputPath = 'config.yaml',
        fcomponents: InputPath = 'components.yaml'
    ) -> Sim:
    """Load YAML inputs, build and run a simulation, and return its driver."""
    with open(f'{path}/{fconfig}','r') as stream:
        config = safe_load(stream)

    with open(f'{path}/{fcomponents}','r') as stream:
        components = safe_load(stream)

    mysim = Sim(path,config,components)
    mysim.build_system()
    mysim.simulate()
    return mysim

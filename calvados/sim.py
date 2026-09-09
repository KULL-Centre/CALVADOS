import os
from datetime import datetime

import mdtraj as md
import numpy as np
import openmm
from Bio.SeqUtils import seq3
from openmm import app, unit
from tqdm import tqdm
from yaml import safe_load

from calvados import build, interactions

from .components import Component, COMPONENT_REGISTRY
from .inputmodels import validate_inputs


def _split_steps(steps: int, max_batches: int = 10) -> list[int]:
    """Split simulation steps into at most ``max_batches`` nonempty batches."""
    nbatches = min(max_batches, steps)
    batch, remainder = divmod(steps, nbatches)
    return [batch + (index < remainder) for index in range(nbatches)]


class Sim:
    def __init__(self,path,config,components):
        """
        simulate openMM Calvados;
        parameters are provided by config dictionary """

        self.path = path
        self.config_model, self.comp_dict = validate_inputs(config, components)

        for key, val in self.config_model.model_dump(mode="python").items():
            setattr(self, key, val)

        self.box = np.array(self.box, dtype=float)
        self.eps_lj *= 4.184 # kcal to kJ/mol

        if self.restart == 'checkpoint' and os.path.isfile(f'{self.path}/{self.frestart}'):
            self.slab_eq = False
            self.bilayer_eq = False
            self.box_eq = False

        if self.slab_eq:
            self.rcent = interactions.init_slab_restraints(self.box,self.k_eq,self.slab_eq_axis)

        if self.ext_force:
            self.rcent = openmm.CustomExternalForce(self.ext_force_expr)

    def make_components(self):
        self.components = np.empty(0)
        self.use_restraints = False

        for name, comp_params in self.comp_dict.items():
            component_class = COMPONENT_REGISTRY[comp_params.molecule_type]
            comp = component_class(name, comp_params)

            # if molecule_type == 'protein':
            #     # Protein component
            
            #     comp = Protein(name, comp_params)
            # elif molecule_type in ['lipid','cooke_lipid']:
            #     # Lipid component
            #     comp = Lipid(name, comp_params)
            # elif molecule_type in ['crowder']:
            #     # Crowder component
            #     comp_setup = 'compact'
            #     comp = Crowder(name, comp_params)
            # elif molecule_type in ['rna']:
            #     # Crowder component
            #     comp_setup = 'spiral'
            #     comp = RNA(name, comp_params)
            # elif molecule_type == 'cyclic':
            #     comp_setup = 'compact'
            #     comp = Cyclic(name, comp_params)
            # elif molecule_type == 'seastar':
            #     comp_setup = 'compact'
            #     comp = Seastar(name, comp_params)
            # elif molecule_type == 'ptm_protein':
            #     comp_setup = 'compact'
            #     comp = PTMProtein(name, comp_params)
            # else:
            #     raise ValueError(f"Component of type {comp_params.molecule_type} not found.")

            comp.eps_lj = self.eps_lj
            comp.calc_properties(pH=self.pH, verbose=self.verbose)
            if comp.restraint:
                if comp.restraint_type == 'go':
                    comp.init_restraint_force(
                        eps_lj=self.eps_lj, cutoff_lj=self.cutoff_lj,
                        cutoff_yu=self.cutoff_yu,
                        eps_yu=self.eps_yu, k_yu = self.k_yu
                    )
                else:
                    comp.init_restraint_force()
                self.use_restraints = True

            self.components = np.append(self.components, comp)

    def count_components(self):
        """ Count components and molecules. """

        self.ncomponents = 0
        self.nmolecules = 0
        self.nmols_per_comp_type: dict[str,int] = {}
        self.comp_types: set[str] = set()

        for comp in self.components:
            self.ncomponents += 1
            self.nmolecules += comp.nmol

        print(f'Total number of components in the system: {self.ncomponents}')
        print(f'Total number of molecules in the system: {self.nmolecules}')

        if ((self.ncomponents > 1) or (self.nmolecules > 1)) and (self.topol in ['single', 'center']):
            raise ValueError("Topol 'single/center' incompatible with multiple molecules.")

        for comp in self.components:
            self.nmols_per_comp_type[comp.molecule_type] = self.nmols_per_comp_type.get(comp.molecule_type, 0) + comp.nmol
            if comp.nmol > 0:
                self.comp_types.add(comp.molecule_type)

    def build_system(self):
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
        self.eps_yu, self.k_yu = interactions.genParamsDH(self.temp,self.ionic)

        # make components
        self.make_components()
        self.count_components()

        self.solute_types = ["protein", "rna", "cyclic", "seastar", "ptm_protein"]
        self.nsolutes = sum(
            self.nmols_per_comp_type.get(comp_type, 0)
            for comp_type in self.solute_types
        )

        # Re-order "solutes" before non-"solutes"
        molecule_types = np.asarray(
            [comp.molecule_type for comp in self.components]
        )
        solute_mask = np.isin(molecule_types, self.solute_types)

        solute_components = self.components[solute_mask]
        non_solute_components = self.components[~solute_mask]

        self.components = np.append(solute_components, non_solute_components)

        # init interactions
        self.ah, self.yu = interactions.init_nonbonded_interactions(
            self.eps_lj,self.cutoff_lj,self.eps_yu,self.k_yu,self.cutoff_yu,self.fixed_lambda
            )
        if "lipid" in self.comp_types:
            self.cos, self.cn = interactions.init_lipid_interactions(
            self.eps_lj,self.eps_yu,self.cutoff_yu,factor=1.9
            )
        if "cooke_lipid" in self.comp_types:
            if "lipid" in self.comp_types:
                raise ValueError(
                    "lipid and cooke_lipid components cannot both be present"
                )
            self.cos, self.cn = interactions.init_lipid_interactions(
            self.eps_lj,self.eps_yu,self.cutoff_yu,factor=3.0
            )

        self.nparticles = 0 # bead counter
        self.grid_counter = 0 # molecule counter for xy and xyz grids

        self.pos = []


        if self.topol == 'slab': # proteins + rna            
            slab_box = np.array(
                [self.box[0], self.box[1], self.slab_width], dtype=float
            )
            self.xyzgrid = build.build_xyzgrid(self.nsolutes, slab_box)
            self.xyzgrid += np.asarray([0,0,self.box[2]/2.-self.slab_width/2.])
            if "crowder" in self.comp_types: # crowder
                crowder_box = np.array(
                    [self.box[0], self.box[1], self.box[2]/2.-self.slab_outer],
                    dtype=float,
                )
                xyzgrid = build.build_xyzgrid(np.ceil(self.nmols_per_comp_type["crowder"]/2.), crowder_box)
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid, axis=0)
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.slab_outer]), axis=0)
        elif self.topol == 'grid':
            self.xyzgrid = build.build_xyzgrid(self.nmolecules,self.box)
        if "lipid" in self.comp_types:
            self.bilayergrid = build.build_xygrid(int(self.nmols_per_comp_type["lipid"]*1.05),self.box)
            if self.nsolutes > 0:
                outer_box = np.array(
                    [self.box[0], self.box[1], self.box[2]/2.-self.box[0]],
                    dtype=float,
                )
                xyzgrid = build.build_xyzgrid(np.ceil(self.nsolutes/2.), outer_box)
                self.xyzgrid = np.append(xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]), axis=0)
        if "cooke_lipid" in self.comp_types:
            self.bilayergrid = build.build_xygrid(int(self.nmols_per_comp_type["cooke_lipid"]*1.05),self.box)
            if self.nsolutes > 0:
                outer_box = np.array(
                    [self.box[0], self.box[1], self.box[2]/2.-self.box[0]],
                    dtype=float,
                )
                xyzgrid = build.build_xyzgrid(np.ceil(self.nsolutes/2.), outer_box)
                self.xyzgrid = np.append(xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]), axis=0)

        for cidx, comp in enumerate(self.components):
            for idx in range(comp.nmol):
                if self.verbose:
                    print(f'Component {cidx}, Molecule {idx}: {comp.name}')
                # particle definitions
                self.add_mdtraj_topol(comp)
                self.add_particles_system(comp.mws)

                # add interactions + restraints
                if comp.molecule_type in ['protein','crowder','cyclic','seastar','ptm_protein','rna']:
                    xs = self.place_molecule(comp)
                elif comp.molecule_type in ['lipid','cooke_lipid']:
                    xs = self.place_bilayer(comp)
                self.add_interactions(comp)

                # add restraints towards box center
                if (self.slab_eq or self.ext_force) and comp.ext_restraint:
                    self.add_ext_restraints(comp)

        if self.custom_restraints:
            self.map_custom_restraints()
            self.add_custom_restraints()

        self.pdb_cg = f'{self.path}/top.pdb'
        self.cif_cg = f'{self.path}/top.cif'
        a = md.Trajectory(self.pos, self.top, 0, self.box, [90,90,90])
        if self.restart != 'pdb': # only save new topology if no system pdb is given
            a.save_pdb(self.pdb_cg)
            a.save_cif(self.cif_cg)

        self.add_forces_to_system()
        self.print_system_summary()

    def add_forces_to_system(self):
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
            if comp.restraint:
                print(f'Number of restraints for comp {comp.name}: {comp.cs.getNumBonds()}')

        # External force
        if self.ext_force:
            self.system.addForce(self.rcent)

        # Equilibration forces
        if self.slab_eq:
            self.system.addForce(self.rcent)

        # Custom forces
        if self.custom_restraints:
            self.system.addForce(self.cres)
            print(f'Number of custom restraints: {self.cres.getNumBonds()}')

        # Barostat force
        if self.box_eq:
            barostat = openmm.openmm.MonteCarloAnisotropicBarostat(
                    [self.pressure[0]*unit.bar,self.pressure[1]*unit.bar,self.pressure[2]*unit.bar],
                    self.temp*unit.kelvin,self.boxscaling_xyz[0],self.boxscaling_xyz[1],
                    self.boxscaling_xyz[2],1000)
            self.system.addForce(barostat)

        # Bilayer eq. force
        if self.bilayer_eq:
            barostat = openmm.openmm.MonteCarloMembraneBarostat(self.pressure[0]*unit.bar,
                    0*unit.bar*unit.nanometer, self.temp*unit.kelvin,
                    openmm.openmm.MonteCarloMembraneBarostat.XYIsotropic,
                    openmm.openmm.MonteCarloMembraneBarostat.ZFixed, 10000)
            self.system.addForce(barostat)

    def print_system_summary(self, write_xml: bool = True):
        """ Print system information and write xml. """

        if write_xml:
            with open(f'{self.path}/{self.sysname}.xml', 'w') as output:
                output.write(openmm.XmlSerializer.serialize(self.system))

        print(f'{self.nparticles} particles in the system')
        print('---------- FORCES ----------')
        print(f'ah: {self.ah.getNumParticles()} particles, {self.ah.getNumExclusions()} exclusions')
        print(f'yu: {self.yu.getNumParticles()} particles, {self.yu.getNumExclusions()} exclusions')
        if self.slab_eq:
            print(f'Equilibration restraints (rcent) towards box center in {self.slab_eq_axis} direction')
            print(f'rcent: {self.rcent.getNumParticles()} restraints')
        if self.bilayer_eq:
            print(f'Equilibration under zero lateral tension')
        if self.box_eq:
            print(f'Equilibration through changes in box side lengths along '+' and '.join(np.array(['X','Y','Z'])[self.boxscaling_xyz]))

    def place_molecule(self, comp: Component, ntries: int = 10000):
        """
        Place proteins based on topology.
        """

        if self.topol == 'slab':
            x0 = self.xyzgrid[self.grid_counter]
            # x0[2] = self.box[2] / 2. # center in z
            xs = x0 + comp.xinit
            self.grid_counter += 1
        elif self.topol == 'grid':
            x0 = self.xyzgrid[self.grid_counter]
            xs = x0 + comp.xinit
            self.grid_counter += 1
        elif self.topol == 'center':
            x0 = self.box * 0.5 # place in center of box
            xs = x0 + comp.xinit
        elif self.topol == 'shift_ref_bead':
            x0 = self.box * 0.5 # place in center of box
            xs = x0 + comp.xinit
            xs -= comp.xinit[self.ref_bead]
        else:
            xs_others = np.array(self.pos, dtype=float)
            xs = build.random_placement(self.box, xs_others, comp.xinit, ntries=ntries)
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return xs # positions of the comp (to be used for restraints)

    def place_bilayer(self, comp: Component, ntries: int = 10000):
        """
        Place proteins based on topology.
        """
        #print('bilayergrid.shape',self.bilayergrid.shape)
        inserted = False
        attempts = 0
        while not inserted and attempts < ntries and self.bilayergrid.size > 0:
            attempts += 1
            xs_others = np.array(self.pos, dtype=float)
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
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return xs # positions of the comp (to be used for restraints)

    def add_bonds(self, comp, offset):
        """ Add bond forces. """

        exclusion_map = comp.add_bonds(offset)
        self.add_exclusions(exclusion_map)

    def add_angles(self, comp, offset):
        """ Add bond forces. """

        exclusion_map = comp.add_angles(offset)
        self.add_exclusions(exclusion_map)

    def add_restraints(self, comp, offset, exclude_nonbonded = True):
        """ Add restraints to single molecule. """

        exclusion_map = comp.add_restraints(offset)
        if exclude_nonbonded: # exclude ah, yu when restraining
            self.add_exclusions(exclusion_map)

    def add_custom_restraints(self, exclude_nonbonded = True):
        exclusion_map = []
        # self.custom_restr_pairs = []
        self.cres = interactions.init_restraints(self.custom_restraint_type)
        for i, j, r, k in self.custom_restr_abs: # i, j, r, k
            self.cres, restr_pair = interactions.add_single_restraint(
                self.cres, self.custom_restraint_type, r, k, i, j)
            # self.custom_restr_pairs.append(restr_pair)
            exclusion_map.append([i,j])
        if exclude_nonbonded: # exclude cres when restraining
            self.add_exclusions(exclusion_map)

    def add_exclusions(self, exclusion_map):
        # exclude LJ, YU for restrained pairs
        for excl in exclusion_map:
            self.ah = interactions.add_exclusion(self.ah, excl[0], excl[1])
            self.yu = interactions.add_exclusion(self.yu, excl[0], excl[1])
            if ("lipid" in self.comp_types) or ("cooke_lipid" in self.comp_types):
                self.cos.addExclusion(excl[0], excl[1])
                self.cn.addExclusion(excl[0], excl[1])

    def add_interactions(self,comp):
        """
        Protein interactions for one molecule of composition comp
        """

        offset = self.nparticles - comp.nbeads # to get indices of current comp in context of system

        # Add Ashbaugh-Hatch
        for sig, lam in zip(comp.sigmas, comp.lambdas):
            if comp.molecule_type in ['lipid', 'cooke_lipid']:
                self.ah.addParticle([sig*unit.nanometer, lam, 0])
            elif comp.molecule_type == 'crowder':
                self.ah.addParticle([sig*unit.nanometer, lam, -1])
            else: # protein, RNA
                self.ah.addParticle([sig*unit.nanometer, lam, 1])
            if ("lipid" in self.comp_types) or ("cooke_lipid" in self.comp_types):
                if comp.molecule_type in ['lipid', 'cooke_lipid']:
                    self.cos.addParticle([sig*unit.nanometer, lam, 0])
                else:
                    self.cos.addParticle([sig*unit.nanometer, lam, 1])
        # Add Debye-Huckel
        for q in comp.qs:
            self.yu.addParticle([q])

        # Add Charge-Nonpolar Interaction
        if ("lipid" in self.comp_types) or ("cooke_lipid" in self.comp_types):
            id_cn = 1 if comp.molecule_type == 'protein' else -1
            for sig, alpha, q in zip(comp.sigmas, comp.alphas, comp.qs):
                self.cn.addParticle([(sig/2)**3, alpha, q, id_cn])

        # Add bonds
        self.add_bonds(comp, offset)

        if comp.molecule_type == 'rna':
            self.add_angles(comp, offset)

        # Add restraints
        if comp.restraint:
            self.add_restraints(comp,offset)

        # write lists
        if self.verbose:
            comp.write_bonds(self.path)
            if comp.restraint:
                comp.write_restraints(self.path)

    def add_ext_restraints(self,comp):
        """ Add external-potential restraints. """

        offset = self.nparticles - comp.nbeads # to get indices of current comp in context of system
        for i in range(0,comp.nbeads):
            self.rcent.addParticle(i+offset)

    def add_mdtraj_topol(self, comp):
        """ Add one molecule to mdtraj topology. """

        # Note: Move this to component eventually.
        chain = self.top.add_chain()

        if comp.molecule_type == 'rna':
            for idx,resname in enumerate(comp.seq):
                res = self.top.add_residue(resname, chain, resSeq=idx+1)
                self.top.add_atom(resname+"P", element=md.element.phosphorus, residue=res)
                self.top.add_atom(resname+"N", element=md.element.nitrogen, residue=res)
            for i in range(comp.nbeads-1):
                for j in range(1,comp.nbeads):
                    if comp.bond_check(i,j):
                        self.top.add_bond(chain.atom(i), chain.atom(j))
        else:
            for idx,resname in enumerate(comp.seq):
                if comp.molecule_type in ['protein','crowder']:
                    resname = comp.residues.loc[resname,'three']
                res = self.top.add_residue(resname, chain, resSeq=idx+1)
                self.top.add_atom('CA', element=md.element.carbon, residue=res)
            for i in range(chain.n_atoms-1):
                for j in range(i+1, chain.n_atoms):
                    if comp.bond_check(i,j):
                        self.top.add_bond(chain.atom(i), chain.atom(j))

    def add_particles_system(self,mws):
        """ Add particles of one molecule to openMM system. """

        for mw in mws:
            self.system.addParticle(mw*unit.amu)

    def map_custom_restraints(self):
        """ Map input format for custom restraints to absolute bead number """
        custom_restr = self.parse_custom_restraints(self.fcustom_restraints)
        total_beads = 0
        for comp in self.components:
            comp.start_bead = int(total_beads)
            total_beads += comp.nmol * comp.nbeads
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
                if not 1 <= copy <= comp.nmol:
                    raise ValueError(
                        f"Custom restraint copy {copy} is outside the valid range "
                        f"1-{comp.nmol} for component {name!r}."
                    )
                if not 1 <= bead <= comp.nbeads:
                    raise ValueError(
                        f"Custom restraint bead {bead} is outside the valid range "
                        f"1-{comp.nbeads} for component {name!r}."
                    )
                x_abs = comp.start_bead + (copy-1)*comp.nbeads + (bead-1)
                crestr.append(x_abs)
            crestr.append(float(r))
            crestr.append(float(k))
            self.custom_restr_abs.append(crestr)

    @staticmethod
    def parse_custom_restraints(fcustom_restraints):
        custom_restraints = []
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

    def simulate(self):
        """ Simulate. """

        fcheck_in = f'{self.path}/{self.frestart}'
        fcheck_out = f'{self.path}/restart.chk'
        append = False

        if self.restart == 'pdb':
            if os.path.isfile(fcheck_in):
                pdb = app.pdbfile.PDBFile(fcheck_in)
            else:
                pdb = app.pdbfile.PDBFile(self.pdb_cg)
        elif self.restart == 'cif':
            if os.path.isfile(fcheck_in):
                pdb = app.pdbxfile.PDBxFile(fcheck_in)
            else:
                pdb = app.pdbxfile.PDBxFile(self.cif_cg)
        else:
            pdb = app.pdbxfile.PDBxFile(self.cif_cg)

        # use langevin integrator
        integrator = openmm.openmm.LangevinMiddleIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
        if self.random_number_seed is not None:
            integrator.setRandomNumberSeed(self.random_number_seed)
        print(integrator.getFriction(),integrator.getTemperature())

        # assemble simulation
        platform = openmm.Platform.getPlatformByName(self.platform)
        if self.platform == 'CPU':
            simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
                platform.setPropertyDefaultValue('DeviceIndex',str(self.gpu_id))
            simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform)
        print('Running on', platform.getName())

        if (os.path.isfile(fcheck_in)) and (self.restart == 'checkpoint'):
            if not os.path.isfile(f'{self.path}/{self.sysname:s}.dcd'):
                raise Exception(f'Did not find {self.path}/{self.sysname:s}.dcd trajectory to append to!')
            append = True
            print(f'Reading check point file {fcheck_in}')
            print(f'Appending trajectory to {self.path}/{self.sysname:s}.dcd')
            print(f'Appending log file to {self.path}/{self.sysname:s}.log')
            simulation.loadCheckpoint(fcheck_in)
        else:
            if self.restart in ['pdb','cif']:
                print(f'Reading in system configuration {self.frestart}')
            elif self.restart == 'checkpoint':
                print(f'No checkpoint file {self.frestart} found: Starting from new system configuration')
            elif self.restart is None:
                print('Starting from new system configuration')

            if os.path.isfile(f'{self.path}/{self.sysname:s}.dcd'): # backup old dcd if not restarting from checkpoint
                now = datetime.now()
                dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")
                print(f'Backing up existing {self.path}/{self.sysname:s}.dcd to {self.path}/backup_{self.sysname:s}_{dt_string}.dcd')
                os.system(f'mv {self.path}/{self.sysname:s}.dcd {self.path}/backup_{self.sysname:s}_{dt_string}.dcd')
            print(f'Writing trajectory to new file {self.path}/{self.sysname:s}.dcd')
            simulation.context.setPositions(pdb.positions)
            print(f'Minimizing energy.')
            simulation.minimizeEnergy()

        if self.slab_eq:
            print(f"Starting slab equilibration with k_eq == {self.k_eq:.4f} kJ/(mol*nm) for {self.steps_eq} steps", flush=True)
            simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/equilibration_{self.sysname:s}.dcd',self.wfreq,append=append))
            simulation.step(self.steps_eq)
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
            integrator = openmm.openmm.LangevinIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
            if self.random_number_seed is not None:
                integrator.setRandomNumberSeed(self.random_number_seed)
            if self.platform == 'CPU':
                simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
            else:
                simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform)
            simulation.context.setPositions(pdb.positions)
            print(f'Minimizing energy.')
            simulation.minimizeEnergy()

        if self.box_eq or self.bilayer_eq:
            print(f"Starting pressure equilibration for {self.steps_eq} steps", flush=True)
            simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/equilibration_{self.sysname:s}.dcd',self.wfreq,append=append))
            simulation.step(self.steps_eq)
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
            if not self.pressure_coupling:
                for index, force in enumerate(self.system.getForces()):
                    if isinstance(force, openmm.openmm.MonteCarloMembraneBarostat):
                        print(f'Removing barostat {index}')
                        self.system.removeForce(index)
                        break
                    if isinstance(force, openmm.openmm.MonteCarloAnisotropicBarostat):
                        print(f'Removing barostat {index}')
                        self.system.removeForce(index)
                        break
            for index, force in enumerate(self.system.getForces()):
                print(index,force)
            integrator = openmm.openmm.LangevinIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
            if self.random_number_seed is not None:
                integrator.setRandomNumberSeed(self.random_number_seed)
            if self.platform == 'CPU':
                simulation = app.simulation.Simulation(topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
            else:
                simulation = app.simulation.Simulation(topology, self.system, integrator, platform)
            simulation.context.setPositions(state_final.getPositions())
            simulation.context.setPeriodicBoxVectors(a, b, c)

        # run simulation
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/{self.sysname:s}.dcd',self.wfreq,append=append))
        simulation.reporters.append(app.statedatareporter.StateDataReporter(f'{self.path}/{self.sysname}.log',self.logfreq,
                step=True,speed=True,elapsedTime=True,potentialEnergy=self.report_potential_energy,separator='\t',append=append))

        print("STARTING SIMULATION", flush=True)
        if self.runtime is not None: # in hours
            simulation.runForClockTime(self.runtime*unit.hour, checkpointFile=fcheck_out, checkpointInterval=30*unit.minute)
        else:
            for batch in tqdm(_split_steps(self.steps), mininterval=1):
                simulation.step(batch)
                simulation.saveCheckpoint(fcheck_out)

        simulation.saveCheckpoint(fcheck_out)

        now = datetime.now()
        dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")

        state_final = simulation.context.getState(getPositions=True,enforcePeriodicBox=True)
        with open(f'{self.path}/checkpoint.pdb', 'w') as f:
            app.PDBFile.writeFile(simulation.topology, state_final.getPositions(), f)
        with open(f'{self.path}/checkpoint.cif', 'w') as f:
            app.PDBxFile.writeFile(simulation.topology, state_final.getPositions(), f)

def run(path='.',fconfig='config.yaml',fcomponents='components.yaml'):
    with open(f'{path}/{fconfig}','r') as stream:
        config = safe_load(stream)

    with open(f'{path}/{fcomponents}','r') as stream:
        components = safe_load(stream)

    mysim = Sim(path,config,components)
    mysim.build_system()
    mysim.simulate()
    return mysim

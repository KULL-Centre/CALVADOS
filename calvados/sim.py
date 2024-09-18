import numpy as np
import openmm
from openmm import app, unit
from pandas import read_csv
from datetime import datetime

import mdtraj as md

from tqdm import tqdm
from importlib import resources
import os

from calvados import build, interactions

from yaml import safe_load

from .components import Component, Protein, Lipid, Crowder
from .build import check_ssdomain

class Sim:
    def __init__(self,path,config,components):
        """
        simulate openMM Calvados;
        parameters are provided by config dictionary """

        self.path = path
        # parse config
        for key, val in config.items():
            setattr(self, key, val)

        for key, val in components['defaults'].items():
            setattr(self, f'default_{key}', val)

        self.comp_dict = components['system']
        self.comp_defaults = components['defaults']

        self.pkg_base = resources.files('calvados')
        f = config.get('fresidues', f'{self.pkg_base}/data/residues.csv')
        self.residues = read_csv(f).set_index('one')

        self.box = np.array(self.box)
        self.eps_lj *= 4.184 # kcal to kJ/mol

        if self.slab_eq:
            self.rcent = interactions.init_eq_restraints(self.box,self.k_eq)

    def make_components(self):
        self.components = np.empty(0)
        self.use_restraints = False
        spiral = True
        for name, properties in self.comp_dict.items():
            if properties.get('molecule_type', self.default_molecule_type) == 'protein':
                # Protein component
                comp = Protein(name, properties, self.comp_defaults)
            elif properties.get('molecule_type', self.default_molecule_type) == 'cooke_lipid':
                # Lipid component
                comp = Lipid(name, properties, self.comp_defaults)
                spiral = False
            elif properties.get('molecule_type', self.default_molecule_type) == 'crowder':
                # Crowder component
                comp = Crowder(name, properties, self.comp_defaults)
            else:
                # Generic component
                comp = Component(name, properties, self.comp_defaults)
            comp.calc_comp_seq(self.ffasta,self.pdb_folder)
            comp.calc_comp_properties(residues=self.residues, pH=self.pH, verbose=self.verbose)
            if comp.restr:
                comp.calc_x_from_pdb(self.pdb_folder, self.use_com)
                comp.calc_dmap()
                self.use_restraints = True
                if self.restraint_type == 'harmonic':
                    comp.calc_ssdomains(self.fdomains)
                elif self.restraint_type == 'go':
                    comp.calc_go_scale(
                            pdb_folder=self.pdb_folder, bfac_width=self.bfac_width,
                            bfac_shift=self.bfac_shift, pae_width=self.pae_width,
                            pae_shift=self.pae_shift, colabfold=self.colabfold)
            else:
                spiral = False if self.topol=='shift_ref_bead' else spiral
                comp.calc_x_linear(spiral=spiral)

            self.components = np.append(self.components, comp)

    def count_components(self):
        """ Count components and molecules. """

        self.ncomponents = 0
        self.nmolecules = 0

        for comp in self.components:
            self.ncomponents += 1
            self.nmolecules += comp.nmol

        print(f'Total number of components in the system: {self.ncomponents}')
        print(f'Total number of molecules in the system: {self.nmolecules}')

        # move lipids at the end of the array
        molecule_types = np.asarray([c.molecule_type for c in self.components])
        self.ncookelipids = np.sum([c.nmol if c.molecule_type == 'cooke_lipid' else 0 for c in self.components])
        self.nproteins = np.sum([c.nmol if c.molecule_type == 'protein' else 0 for c in self.components])
        self.ncrowders = np.sum([c.nmol if c.molecule_type == 'crowder' else 0 for c in self.components])

        if ((self.ncomponents > 1) or (self.nmolecules > 1)) and (self.topol in ['single', 'center']):
            raise ValueError("Topol 'center' incompatible with multiple molecules.")

        # move proteins at the beginning of the array
        if self.nmolecules > self.nproteins:
            protein_components = self.components[np.where(molecule_types=='protein')]
            non_protein_components = self.components[np.where(molecule_types!='protein')]
            self.components = np.append(protein_components,non_protein_components)

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

        # make components
        self.make_components()
        self.count_components()

        self.bond_pairlist = []

        # init interaction parameters
        self.eps_yu, self.k_yu = interactions.genParamsDH(self.temp,self.ionic)

        # init restraints
        if self.use_restraints:
            self.cs = interactions.init_restraints(self.restraint_type)
            self.restr_pairlist = []
            if self.restraint_type == 'go':
                self.scLJ_pairlist = []
                self.scYU_pairlist = []
                self.scLJ = interactions.init_scaled_LJ(self.eps_lj,self.cutoff_lj)
                self.scYU = interactions.init_scaled_YU(self.eps_yu,self.k_yu)

        # init interactions
        self.hb, self.ah, self.yu = interactions.init_protein_interactions(
            self.eps_lj,self.cutoff_lj,self.eps_yu,self.k_yu,self.cutoff_yu,self.fixed_lambda
            )
        if self.ncookelipids > 0:
            self.wcafene, self.cos, self.cn = interactions.init_cooke_lipid_interactions(
            self.eps_lj,self.eps_yu,self.cutoff_yu
            )

        self.nparticles = 0 # bead counter
        self.grid_counter = 0 # molecule counter for xy and xyz grids

        self.pos = []
        # self.make_sys_dataframe() # pool component information into one df
        # print(self.df_sys, flush=True)
        if self.topol == 'slab':
            self.xyzgrid = build.build_xyzgrid(self.nproteins,[self.box[0],self.box[1],2*self.box[0]])
            self.xyzgrid += np.asarray([0,0,self.box[2]/2.-self.box[0]])
            if self.ncrowders > 0:
                xyzgrid = build.build_xyzgrid(np.ceil(self.ncrowders/2.),[self.box[0],self.box[1],self.box[2]/2.-2*self.box[0]])
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid, axis=0)
                self.xyzgrid = np.append(self.xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+2*self.box[0]]), axis=0)
        elif self.topol == 'grid':
            self.xyzgrid = build.build_xyzgrid(self.nmolecules,self.box)
        if self.ncookelipids > 0:
            self.bilayergrid = build.build_xygrid(int(self.ncookelipids*1.05),self.box)
            if self.nproteins > 0:
                xyzgrid = build.build_xyzgrid(np.ceil(self.nproteins/2.),[self.box[0],self.box[1],self.box[2]/2.-self.box[0]])
                self.xyzgrid = np.append(xyzgrid, xyzgrid + np.asarray([0,0,self.box[2]/2.+self.box[0]]), axis=0)

        for cidx, comp in enumerate(self.components):

            for idx in range(comp.nmol):
                if self.verbose:
                    print(f'Component {cidx}, Molecule {idx}: {comp.name}')
                # particle definitions
                self.add_mdtraj_topol(comp.seq)
                self.add_particles_system(comp.mws)

                # add interactions + restraints
                if comp.molecule_type == 'protein':
                    xs = self.place_molecule(comp)
                elif comp.molecule_type == 'crowder':
                    xs = self.place_molecule(comp)
                elif comp.molecule_type == 'cooke_lipid':
                    xs = self.place_bilayer(comp)
                elif comp.molecule_type == 'rna':
                    raise
                self.add_interactions(comp)

                # add restraints towards box center
                if self.slab_eq and comp.molecule_type == 'protein':
                    self.add_eq_restraints(comp)

        self.pdb_cg = f'{self.path}/top.pdb'
        a = md.Trajectory(self.pos, self.top, 0, self.box, [90,90,90])
        if self.restart != 'pdb': # only save new topology if no system pdb is given
            a.save_pdb(self.pdb_cg)

        self.add_forces_to_system()
        self.print_system_summary()

    def add_forces_to_system(self):
        """ Add forces to system. """

        for force in [self.hb, self.yu, self.ah]:
            self.system.addForce(force)
        if self.ncookelipids > 0:
            for force in [self.wcafene, self.cos, self.cn]:
                self.system.addForce(force)
        if self.use_restraints:
            self.system.addForce(self.cs)
            print(f'Number of restraints: {self.cs.getNumBonds()}')

            if (self.restraint_type == 'go'):
                self.system.addForce(self.scLJ)
                self.system.addForce(self.scYU)
                # print(f'Number of scaled LJ pairs: {self.scLJ.getNumBonds()}')
                # print(f'Number of scaled YU pairs: {self.scYU.getNumBonds()}')
        if self.slab_eq:
            self.system.addForce(self.rcent)
        if self.bilayer_eq:
            barostat = openmm.openmm.MonteCarloMembraneBarostat(0*unit.bar,
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
            print(f'Equilibration restraints (rcent) towards box center in z direction')
            print(f'rcent: {self.rcent.getNumParticles()} restraints')
        if self.bilayer_eq:
            print(f'Equilibration under zero lateral tension')

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
            xs = build.random_placement(self.box, self.pos, comp.xinit, ntries=ntries)
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
        while not inserted:
            xs, inserted = build.build_xybilayer(self.bilayergrid[0], self.box, self.pos, comp.xinit)
            if not inserted:
                xs, inserted = build.build_xybilayer(self.bilayergrid[0], self.box, self.pos, comp.xinit, upward=False)
                idx = np.random.randint(self.bilayergrid.shape[0])
                self.bilayergrid[0] = self.bilayergrid[idx]
                self.bilayergrid = np.delete(self.bilayergrid,idx,axis=0)
        for x in xs:
            self.pos.append(x)
            self.nparticles += 1
        return xs # positions of the comp (to be used for restraints)

    def add_bonds(self, comp, offset):
        """ Add bond forces. """
        for i in range(0,comp.nres-1):
            for j in range(i, comp.nres):
                if comp.bond_map[i,j]:
                    if comp.molecule_type in ["protein", "crowder"]:
                        d0 = 0.5 * (comp.bondlengths[i] + comp.bondlengths[j])
                        if comp.restr:
                            if self.restraint_type == 'harmonic':
                                ss = build.check_ssdomain(comp.ssdomains,i,j,req_both=False)
                                d = comp.dmap[i,j] if ss else d0
                            elif self.restraint_type == 'go':
                                # bondscale = max(0, 1. - 5.*comp.scale[i,j])
                                d = comp.bondscale[i,j] * d0 + (1. - comp.bondscale[i,j]) * comp.dmap[i,j]
                                # d = comp.scale[i,j] * comp.dmap[i,j] + (1. - comp.scale[i,j]) * d0
                        else:
                            d = d0
                        bidx = self.hb.addBond(i+offset, j+offset, d*unit.nanometer, self.kb*unit.kilojoules_per_mole/(unit.nanometer**2))
                        self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,d,self.kb]) # 1-based
                        self.yu.addExclusion(i+offset, j+offset)
                        self.ah.addExclusion(i+offset, j+offset)
                        if self.ncookelipids > 0:
                            self.cos.addExclusion(i+offset, j+offset)
                            self.cn.addExclusion(i+offset, j+offset)

                    if comp.molecule_type == "cooke_lipid":
                        if j-i == 1:
                            d = 0.5 * (comp.bondlengths[i] + comp.bondlengths[j])
                            kfene = 20*3*self.eps_lj/d/d
                            bidx = self.wcafene.addBond(i+offset, j+offset, [d*unit.nanometer, kfene*unit.kilojoules_per_mole/(unit.nanometer**2)])
                            self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,d,kfene]) # 1-based

                            self.ah.addExclusion(i+offset, j+offset)
                            self.yu.addExclusion(i+offset, j+offset)
                            self.cos.addExclusion(i+offset, j+offset)
                            self.cn.addExclusion(i+offset, j+offset)
                        else:
                            d = 0.5 * (comp.bondlengths[i] + comp.bondlengths[j])
                            kbend = 20*self.eps_lj/d/d
                            bidx = self.hb.addBond(i+offset, j+offset, 4*d*unit.nanometer, kbend*unit.kilojoules_per_mole/(unit.nanometer**2))
                            self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,4*d,kbend]) # 1-based

                # # Add weakened LJ, full YU, if restraint is weak
                # if comp.scale[i,j] < 0.98:
                #     self.scLJ.addBond(i+offset,j+offset, [s*unit.nanometer, l*unit.dimensionless, n*unit.dimensionless])
                #     qij = qi * qj * self.eps_yu * self.eps_yu *unit.nanometer*unit.kilojoules_per_mole *unit.nanometer * unit.kilojoules_per_mole
                #     self.scYU.addBond(i+offset,j+offset, [qij, n*unit.dimensionless])
                #     scaled_LJYU_pairlist.append([i+offset+1,j+offset+1, n])

    def add_restraints(self, comp, offset, min_scale = 0.02):
        """ Add restraints to single molecule. """
        # restr_pairlist = []
        for i in range(0,comp.nres-2):
            # qi = comp.qs[i]
            for j in range(i+2,comp.nres):
                # check if below cutoff
                if comp.dmap[i,j] > self.cutoff_restr:
                    continue
                # harmonic
                if self.restraint_type == 'harmonic':
                    ss = check_ssdomain(comp.ssdomains,i,j,req_both=True)
                    if not ss:
                        continue
                    k = self.k_harmonic
                # go
                elif self.restraint_type == 'go':
                    if comp.scale[i,j] < min_scale:
                        continue
                    k = self.k_go * comp.scale[i,j]
                    # add scaled pseudo LJ, YU for low restraints
                    if comp.bondscale[i,j] > min_scale:
                        self.scLJ, scaled_pair = interactions.add_scaled_lj(self.scLJ, i, j, offset, comp)
                        self.scLJ_pairlist.append(scaled_pair)
                        if comp.qs[i] * comp.qs[j] != 0.:
                            self.scYU, scaled_pair = interactions.add_scaled_yu(self.scYU, i, j, offset, comp)
                            self.scYU_pairlist.append(scaled_pair)

                self.cs, restr_pair = interactions.add_single_restraint(
                        self.cs, self.restraint_type, comp.dmap[i,j], k,
                        i+offset, j+offset)
                self.restr_pairlist.append(restr_pair)

                # exclude LJ, YU for restrained pairs
                self.ah = interactions.add_exclusion(self.ah, i+offset, j+offset)
                self.yu = interactions.add_exclusion(self.yu, i+offset, j+offset)
                if self.ncookelipids > 0:
                    self.cos.addExclusion(i+offset, j+offset)
                    self.cn.addExclusion(i+offset, j+offset)

        # return restr_pairlist

    def add_interactions(self,comp):
        """
        Protein interactions for one molecule of composition comp
        """

        offset = self.nparticles - comp.nres # to get indices of current comp in context of system

        # Add Ashbaugh-Hatch
        for sig, lam in zip(comp.sigmas, comp.lambdas):
            if comp.molecule_type == 'cooke_lipid':
                self.ah.addParticle([sig*unit.nanometer, lam, 0])
            elif comp.molecule_type == 'crowder':
                self.ah.addParticle([sig*unit.nanometer, lam, -1])
            else:
                self.ah.addParticle([sig*unit.nanometer, lam, 1])
            if self.ncookelipids > 0:
                if comp.molecule_type == 'cooke_lipid':
                    self.cos.addParticle([sig*unit.nanometer, lam, 1])
                else:
                    self.cos.addParticle([sig*unit.nanometer, lam, 0])

        # Add Debye-Huckel
        for q in comp.qs:
            self.yu.addParticle([q])

        # Add Charge-Nonpolar Interaction
        if self.ncookelipids > 0:
            id_cn = 1 if comp.molecule_type == 'protein' else -1
            for sig, alpha, q in zip(comp.sigmas, comp.alphas, comp.qs):
                self.cn.addParticle([(sig/2)**3, alpha, q, id_cn])

        # Add bonds
        self.add_bonds(comp,offset)

        # Add restraints
        if comp.restr:
            self.add_restraints(comp,offset)

        # write lists
        if self.verbose:
            self.write_bonds()
            if comp.restr:
                self.write_restraints(comp.name)
                # if self.restraint_type == 'go':
                    # self.write_scaled_LJYU(comp,scaled_LJYU_pairlist)

    def write_bonds(self):
        """ Write bonds to file. """

        with open(f'{self.path}/bonds.txt','w') as f:
            f.write('i\tj\tb_idx\td[nm]\tk[kJ/mol/nm^2]\n')
            for b in self.bond_pairlist:
                f.write(f'{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{b[3]:.4f}\t{b[4]:.4f}\n')

    def write_restraints(self,compname):
        """ Write restraints to file. """

        with open(f'{self.path}/restr.txt','w') as f:
            f.write('i j d[nm] fc\n')
            for r in self.restr_pairlist:
                f.write(f'{int(r[0])} {int(r[1])} {r[2]:.4f} {r[3]:.4f}\n')

        if self.restraint_type == 'go':
            with open(f'{self.path}/scaled_LJ_{compname}.txt','w') as f:
                f.write('i+offset+1, j+offset+1, s, l, comp.bondscale[i,j]\n') # 1-based
                for r in self.scLJ_pairlist:
                    f.write(f'{int(r[0])} {int(r[1])} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f}\n')

            with open(f'{self.path}/scaled_YU_{compname}.txt','w') as f:
                f.write('i+offset+1, j+offset+1, comp.bondscale[i,j]\n') # 1-based
                for r in self.scYU_pairlist:
                    f.write(f'{int(r[0])} {int(r[1])} {r[2]:.4f}\n')
    # def write_scaled_LJYU(self,comp,scaled_LJYU_pairlist):
    #     with open(f'{self.path}/scaled_LJYU_{comp.name}_idx.txt','w') as f:
    #         f.write('i j scaling\n')
    #         for r in scaled_LJYU_pairlist:
    #             f.write(f'{int(r[0])} {int(r[1])} {r[2]:.4f}\n')

    def add_eq_restraints(self,comp):
        """ Add equilibration restraints. """

        offset = self.nparticles - comp.nres # to get indices of current comp in context of system
        for i in range(0,comp.nres):
            self.rcent.addParticle(i+offset)

    def add_mdtraj_topol(self,seq):
        """ Add one molecule to mdtraj topology. """

        chain = self.top.add_chain()
        for idx,resname in enumerate(seq):
            res = self.top.add_residue(self.residues.loc[resname].three, chain, resSeq=idx+1)
            self.top.add_atom('CA', element=md.element.carbon, residue=res)
        for i in range(chain.n_atoms-1):
            self.top.add_bond(chain.atom(i),chain.atom(i+1))

    def add_particles_system(self,mws):
        """ Add particles of one molecule to openMM system. """

        for mw in mws:
            self.system.addParticle(mw*unit.amu)

    def simulate(self):
        """ Simulate. """

        if self.restart == 'pdb':
            pdb = app.pdbfile.PDBFile(self.frestart)
        else:
            pdb = app.pdbfile.PDBFile(self.pdb_cg)

        # use langevin integrator
        integrator = openmm.openmm.LangevinIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
        print(integrator.getFriction(),integrator.getTemperature())

        # assemble simulation
        platform = openmm.Platform.getPlatformByName(self.platform)
        if self.platform == 'CPU':
            print('Running on', platform.getName())
            simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
        else:
            if os.environ.get('CUDA_VISIBLE_DEVICES') == None:
                platform.setPropertyDefaultValue('DeviceIndex',str(self.gpu_id))
            print('Running on', platform.getName())
            simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform)

        fcheck_in = f'{self.path}/{self.frestart}'
        fcheck_out = f'{self.path}/restart.chk'
        append = False

        if (os.path.isfile(fcheck_in)) and (self.restart == 'checkpoint'):
            if not os.path.isfile(f'{self.path}/{self.sysname:s}.dcd'):
                raise Exception(f'Did not find {self.path}/{self.sysname:s}.dcd trajectory to append to!')
            append = True
            print(f'Reading check point file {fcheck_in}')
            print(f'Appending trajectory to {self.path}/{self.sysname:s}.dcd')
            print(f'Appending log file to {self.path}/{self.sysname:s}.log')
            simulation.loadCheckpoint(fcheck_in)
        else:
            if self.restart == 'pdb':
                print(f'Reading in system configuration {self.frestart}')
            elif self.restart == 'checkpoint':
                print(f'No checkpoint file {self.frestart} found: Starting from new system configuration')
            elif self.restart == None:
                print('Starting from new system configuration')
            else:
                raise

            if os.path.isfile(f'{self.path}/{self.sysname:s}.dcd'): # backup old dcd if not restarting from checkpoint
                now = datetime.now()
                dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")
                print(f'Backing up existing {self.path}/{self.sysname:s}.dcd to {self.path}/backup_{self.sysname:s}_{dt_string}.dcd')
                os.system(f'mv {self.path}/{self.sysname:s}.dcd {self.path}/backup_{self.sysname:s}_{dt_string}.dcd')
            print(f'Writing trajectory to new file {self.path}/{self.sysname:s}.dcd')
            simulation.context.setPositions(pdb.positions)
            print(f'Minimizing energy.')
            simulation.minimizeEnergy()

        if self.slab_eq and not os.path.isfile(fcheck_in):
            print(f"Starting equilibration with k_eq == {self.k_eq:.4f} kJ/(mol*nm) for {self.steps_eq} steps", flush=True)
            simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/equilibration_{self.sysname:s}.dcd',self.wfreq,append=append))
            simulation.step(self.steps_eq)
            state_final = simulation.context.getState(getPositions=True)
            rep = app.pdbreporter.PDBReporter(f'{self.path}/equilibration_final.pdb',0)
            rep.report(simulation,state_final)
            pdb = app.pdbfile.PDBFile(f'{self.path}/equilibration_final.pdb')

            for index, force in enumerate(self.system.getForces()):
                if isinstance(force, openmm.CustomExternalForce):
                    print(f'Removing external force {index}')
                    self.system.removeForce(index)
                    break
            integrator = openmm.openmm.LangevinIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
            if self.platform == 'CPU':
                simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
            else:
                simulation = app.simulation.Simulation(pdb.topology, self.system, integrator, platform)
            simulation.context.setPositions(pdb.positions)
            print(f'Minimizing energy.')
            simulation.minimizeEnergy()

        if self.bilayer_eq and not os.path.isfile(fcheck_in):
            print(f"Starting equilibration under zero lateral tension for {self.steps_eq} steps", flush=True)
            simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/equilibration_{self.sysname:s}.dcd',self.wfreq,append=append))
            simulation.step(self.steps_eq)
            state_final = simulation.context.getState(getPositions=True,enforcePeriodicBox=True)
            rep = app.pdbreporter.PDBReporter(f'{self.path}/equilibration_final.pdb',0)
            rep.report(simulation,state_final)
            pdb = app.pdbfile.PDBFile(f'{self.path}/equilibration_final.pdb')
            topology = pdb.getTopology()
            a, b, c = state_final.getPeriodicBoxVectors()
            topology.setPeriodicBoxVectors(state_final.getPeriodicBoxVectors())
            for index, force in enumerate(self.system.getForces()):
                print(index,force)
            if not self.zero_lateral_tension:
                for index, force in enumerate(self.system.getForces()):
                    if isinstance(force, openmm.openmm.MonteCarloMembraneBarostat):
                        print(f'Removing membrane barostat {index}')
                        self.system.removeForce(index)
                        break
            for index, force in enumerate(self.system.getForces()):
                print(index,force)
            integrator = openmm.openmm.LangevinIntegrator(self.temp*unit.kelvin,self.friction_coeff/unit.picosecond,0.01*unit.picosecond)
            if self.platform == 'CPU':
                simulation = app.simulation.Simulation(topology, self.system, integrator, platform, dict(Threads=str(self.threads)))
            else:
                simulation = app.simulation.Simulation(topology, self.system, integrator, platform)
            simulation.context.setPositions(state_final.getPositions())
            simulation.context.setPeriodicBoxVectors(a, b, c)
            #print(f'Minimizing energy.')
            #simulation.minimizeEnergy()

        # run simulation
        simulation.reporters.append(app.dcdreporter.DCDReporter(f'{self.path}/{self.sysname:s}.dcd',self.wfreq,append=append))
        simulation.reporters.append(app.statedatareporter.StateDataReporter(f'{self.path}/{self.sysname}.log',int(self.wfreq*10),
                step=True,speed=True,elapsedTime=True,separator='\t',append=append))

        print("STARTING SIMULATION", flush=True)
        if self.runtime > 0: # in hours
            simulation.runForClockTime(self.runtime*unit.hour, checkpointFile=fcheck_out, checkpointInterval=30*unit.minute)
        else:
            nbatches = 10
            batch = int(self.steps / nbatches)
            for i in tqdm(range(nbatches),mininterval=1):
                simulation.step(batch)
                simulation.saveCheckpoint(fcheck_out)
        simulation.saveCheckpoint(fcheck_out)

        now = datetime.now()
        dt_string = now.strftime("%Y%d%m_%Hh%Mm%Ss")

        state_final = simulation.context.getState(getPositions=True,enforcePeriodicBox=True)
        rep = app.pdbreporter.PDBReporter(f'{self.path}/{self.sysname}_{dt_string}.pdb',0)
        rep.report(simulation,state_final)
        rep = app.pdbreporter.PDBReporter(f'{self.path}/checkpoint.pdb',0)
        rep.report(simulation,state_final)

def run(path='.',fconfig='config.yaml',fcomponents='components.yaml'):
    with open(f'{path}/{fconfig}','r') as stream:
        config = safe_load(stream)

    with open(f'{path}/{fcomponents}','r') as stream:
        components = safe_load(stream)

    mysim = Sim(path,config,components)
    mysim.build_system()
    mysim.simulate()
    return mysim

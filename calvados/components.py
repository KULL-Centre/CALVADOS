from .sequence import seq_from_pdb, read_fasta, get_qs, patch_terminal_qs
from .analysis import self_distances
from calvados import build, interactions

from pandas import read_csv

from openmm import unit

import numpy as np

class Component:
    """ Generic component of the system. """

    def __init__(self, name: str, properties: dict, defaults: dict):
        self.name = name

        # read component properties
        for key, val in properties.items():
            setattr(self, key, val)

        # read default properties where necessary
        for key, val in defaults.items():
            if key not in properties:
                setattr(self, key, val)

        # read residue parameters from file
        try:
            self.residues = read_csv(self.fresidues).set_index('one')
        except AttributeError:
            raise FileNotFoundError(f'Residue parameter file name (fresidues) not supplied to component {name}.')

    def calc_comp_seq(self):
        """ Calculate sequence of component. """

        if self.restraint:
            self.seq, self.n_termini, self.c_termini = seq_from_pdb(f'{self.pdb_folder}/{self.name}.pdb')
        else:
            records = read_fasta(self.ffasta)
            self.seq = str(records[self.name].seq)
            self.n_termini = [0]
            self.c_termini = [len(self.seq)-1]

    def calc_properties(self, pH: float = 7.0, verbose: bool = False):
        """ Calculate component properties (sigmas, lambdas, qs etc.) """

        self.calc_comp_seq()
        self.nres = len(self.seq)
        self.nbeads = self.nres
        self.sigmas = np.array([self.residues.loc[s].sigmas for s in self.seq])
        self.lambdas = np.array([self.residues.loc[s].lambdas for s in self.seq])
        self.bondlengths = np.array([self.residues.loc[s].bondlength for s in self.seq])
        self.mws = np.array([self.residues.loc[s].MW for s in self.seq])
        self.qs, _ = get_qs(self.seq,flexhis=True,pH=pH,residues=self.residues)
        self.alphas = self.lambdas*self.alpha
        self.init_bond_force()

    def calc_dmap(self):
        if self.periodic:
            self.dmap = self_distances(self.xinit,self.dimensions)
        else:
            self.dmap = self_distances(self.xinit)

    def calc_x_setup(self, d: float = 0.38, comp_setup: str = 'spiral',
            n_per_res: int = 1, ys = None):
        if comp_setup == 'spiral':
            self.xinit = build.build_spiral(self.bondlengths, arc=d, n_per_res=n_per_res)
        elif comp_setup == 'compact':
            self.xinit = build.build_compact(self.nbeads, d=d)
        else:
            self.xinit = build.build_linear(self.bondlengths, n_per_res=n_per_res, ys=ys)

    @staticmethod
    def bond_check(i: int, j: int):
        """ Placeholder for molecule-specific method. """
        return False

    def calc_bondlength(self, i: int, j: int):
        d0 = 0.5 * (self.bondlengths[i] + self.bondlengths[j])
        return d0

    def init_bond_force(self):
        self.bond_pairlist = []
        self.hb = interactions.init_bonded_interactions()

    def add_bonds(self, offset):
        exclusion_map = [] # for ah, yu etc.
        for i in range(0,self.nbeads-1):
            for j in range(i, self.nbeads):
                if self.bond_check(i,j):
                    d = self.calc_bondlength(i, j)
                    bidx = self.hb.addBond(
                        i+offset, j+offset, d*unit.nanometer,
                        self.kb*unit.kilojoules_per_mole/(unit.nanometer**2))
                    self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,d,self.kb]) # 1-based
                    exclusion_map.append([i+offset,j+offset])
        return exclusion_map

    def get_forces(self):
        self.forces = [self.hb]

    def write_bonds(self, path):
        """ Write bonds to file. """

        with open(f'{path}/bonds_{self.name}.txt','w') as f:
            f.write('i\tj\tb_idx\td[nm]\tk[kJ/mol/nm^2]\n')
            for b in self.bond_pairlist:
                f.write(f'{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{b[3]:.4f}\t{b[4]:.4f}\n')

class Protein(Component):
    """ Component protein. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def calc_x_from_pdb(self):
        """ Calculate protein positions from pdb. """

        input_pdb = f'{self.pdb_folder}/{self.name}.pdb'
        self.xinit, self.dimensions = build.geometry_from_pdb(input_pdb,use_com=self.use_com) # read from pdb

    def calc_ssdomains(self):
        """ Get bounds for restraints (harmonic). """

        self.ssdomains = build.get_ssdomains(self.name,self.fdomains)
        # print(f'Setting {self.restraint_type} restraints for {comp.name}')

    def calc_go_scale(self, bscale_shift = 0.1, bscale_width = 80):#,
            # pdb_folder: str, bfac_width: float = 50., bfac_shift: float = 0.8,
            # pae_width: float = 8., pae_shift: float = 0.4, colabfold: int = 0):
        """ Calculate scaling of Go potential for all residue pairs. """

        input_pdb = f'{self.pdb_folder}/{self.name}.pdb'
        bfac = build.bfac_from_pdb(input_pdb,confidence=0.)
        bfac_map = np.add.outer(bfac,bfac) / 2.
        bfac_sigm_factor = self.bfac_width*(bfac_map-self.bfac_shift)
        bfac_sigm = np.exp(bfac_sigm_factor) / (np.exp(bfac_sigm_factor) + 1)

        input_pae = f'{self.pdb_folder}/{self.name}.json'
        pae = build.load_pae(input_pae,symmetrize=True,colabfold=self.colabfold) / 10. # in nm
        pae_sigm_factor = -self.pae_width*(pae-self.pae_shift)
        pae_sigm = np.exp(pae_sigm_factor) / (np.exp(pae_sigm_factor) + 1)
        # pae_inv = build.load_pae_inv(input_pae,colabfold=self.colabfold)
        # scaled_LJYU_pairlist = []
        self.scale = bfac_sigm * pae_sigm # restraint scale
        # self.bondscale = np.minimum(1.,np.maximum(0, 1. - 10.* (self.scale - min_scale))) # residual interactions for low restraints
        self.bondscale = np.exp(-bscale_width*(self.scale-bscale_shift)) / (np.exp(-bscale_width*(self.scale-bscale_shift)) + 1)
        self.curate_bondscale()

    def curate_bondscale(self, max_bscale = 0.95):
        for idx in range(self.nbeads):
            lower = self.bondscale[idx,:idx-3]
            higher = self.bondscale[idx,idx+4:]
            if idx <= 3:
                others = higher
            elif idx >= self.nbeads - 4:
                others = lower
            else:
                others = np.concatenate([lower, higher])
            count = np.sum(np.where(others < max_bscale,1,0)) # count how many nonlocal restraints
            # print(idx,count)
            if count == 0:
                for jdx in range(max(0,idx-3),min(self.nbeads,idx+4)):
                    self.bondscale[idx,jdx] = 1 # remove local restraints
                    self.bondscale[jdx,idx] = 1 # remove local restraints

    def calc_properties(self, pH: float = 7.0, verbose: bool = False, comp_setup: str = 'spiral'):
        """ Protein properties. """

        super().calc_properties(pH=pH, verbose=verbose)
        # fix mass of termini
        self.mws[self.n_termini] += 2
        self.mws[self.c_termini] += 16
        # fix charge of termini
        if verbose:
            print(f'Adding charges for {self.charge_termini} termini of {self.name}.', flush=True)
        self.qs = patch_terminal_qs(self.qs,self.n_termini,self.c_termini,loc=self.charge_termini)

        if self.restraint:
            # self.init_restraint_force() # Done via sim.py
            self.calc_x_from_pdb()
            self.calc_dmap()
            if self.restraint_type == 'harmonic':
                self.calc_ssdomains()
            elif self.restraint_type == 'go':
                self.calc_go_scale()
        else:
            self.calc_x_setup(comp_setup = comp_setup)

    def calc_bondlength(self, i, j, min_bscale = 0.05, max_bscale = 0.95):
        d0 = 0.5 * (self.bondlengths[i] + self.bondlengths[j])
        if self.restraint:
            if self.restraint_type == 'harmonic':
                ss = build.check_ssdomain(self.ssdomains,i,j,req_both=False)
                d = self.dmap[i,j] if ss else d0
            elif self.restraint_type == 'go':
                if self.bondscale[i,j] > max_bscale:
                    d = d0
                elif self.bondscale[i,j] < min_bscale:
                    d = self.dmap[i,j]
                else:
                    d = self.bondscale[i,j] * d0 + (1. - self.bondscale[i,j]) * self.dmap[i,j]
            else:
                raise ValueError("Restraint type must be harmonic or go.")
        else:
            d = d0
        # print(i,j,self.bondscale[i,j],d0,d)
        return d

    def bond_check(self, i: int, j: int):
        """ Define bonded term conditions. """

        condition = (j == i+1)
        condition_termini = (i not in self.c_termini) and (j not in self.n_termini)
        return condition and condition_termini

    def add_bonds(self, offset):
        exclusion_map = [] # for ah, yu etc.
        for i in range(0,self.nbeads-1):
            for j in range(i, self.nbeads):
                if self.bond_check(i,j):
                    d = self.calc_bondlength(i, j)
                    bidx = self.hb.addBond(
                        i+offset, j+offset, d*unit.nanometer,
                        self.kb*unit.kilojoules_per_mole/(unit.nanometer**2))
                    self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,d,self.kb]) # 1-based
                    exclusion_map.append([i+offset,j+offset])
        return exclusion_map

    def init_restraint_force(self, eps_lj=None, cutoff_lj=None, eps_yu=None, k_yu=None):
        self.cs = interactions.init_restraints(self.restraint_type)
        self.restr_pairlist = []
        if self.restraint_type == 'go':
            self.scLJ_pairlist = []
            self.scYU_pairlist = []
            self.scLJ = interactions.init_scaled_LJ(eps_lj,cutoff_lj)
            self.scYU = interactions.init_scaled_YU(eps_yu,k_yu)

    def add_restraints(self, offset, min_bscale = 0.05, max_bscale = 0.95):
        """ Add restraints. """
        exclusion_map = [] # for ah, yu etc.
        for i in range(0,self.nbeads-2):
            for j in range(i+2,self.nbeads):
                # check if below cutoff
                if self.dmap[i,j] > self.cutoff_restr:
                    continue
                # harmonic
                if self.restraint_type == 'harmonic':
                    ss = build.check_ssdomain(self.ssdomains,i,j,req_both=True)
                    if not ss:
                        continue
                    k = self.k_harmonic
                # go
                elif self.restraint_type == 'go':
                    if self.bondscale[i,j] > max_bscale:
                        continue
                    k = self.k_go * self.scale[i,j]
                    # add scaled pseudo LJ, YU for low restraints
                    if self.bondscale[i,j] > min_bscale:
                        self.scLJ, scaled_pair = interactions.add_scaled_lj(self.scLJ, i, j, offset, self)
                        self.scLJ_pairlist.append(scaled_pair)
                        if self.qs[i] * self.qs[j] != 0.:
                            self.scYU, scaled_pair = interactions.add_scaled_yu(self.scYU, i, j, offset, self)
                            self.scYU_pairlist.append(scaled_pair)

                self.cs, restr_pair = interactions.add_single_restraint(
                        self.cs, self.restraint_type, self.dmap[i,j], k,
                        i+offset, j+offset)
                self.restr_pairlist.append(restr_pair)
                exclusion_map.append([i+offset,j+offset])
        return exclusion_map

    def write_restraints(self,path):
        """ Write restraints to file. """

        with open(f'{path}/restr_{self.name}.txt','w') as f:
            f.write('i j d[nm] fc\n')
            for r in self.restr_pairlist:
                f.write(f'{int(r[0])} {int(r[1])} {r[2]:.4f} {r[3]:.4f}\n')

        if self.restraint_type == 'go':
            with open(f'{path}/scaled_LJ_{self.name}.txt','w') as f:
                f.write('i+offset+1, j+offset+1, s, l, comp.bondscale[i,j]\n') # 1-based
                for r in self.scLJ_pairlist:
                    f.write(f'{int(r[0])} {int(r[1])} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f}\n')

            with open(f'{path}/scaled_YU_{self.name}.txt','w') as f:
                f.write('i+offset+1, j+offset+1, comp.bondscale[i,j]\n') # 1-based
                for r in self.scYU_pairlist:
                    f.write(f'{int(r[0])} {int(r[1])} {r[2]:.4f}\n')

    def get_forces(self):
        self.forces = [self.hb]
        if self.restraint:
            self.forces.append(self.cs)
            if self.restraint_type == 'go':
                self.forces.extend([self.scLJ, self.scYU])

class RNA(Component):
    """ Component RNA. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def calc_properties(self, pH: float = 7.0, verbose: bool = False, comp_setup: str = 'spiral'):
        """ Calculate component properties (sigmas, lambdas, qs etc.) """

        self.calc_comp_seq() # --> seq and seq2
        self.nres = len(self.seq)
        self.nbeads = len(self.seq2)

        self.sigmas = np.array([self.residues.loc[s].sigmas for s in self.seq2])
        self.lambdas = np.array([self.residues.loc[s].lambdas for s in self.seq2])
        self.bondlengths = np.array([self.residues.loc[s].bondlength for s in self.seq2])
        self.mws = np.array([self.residues.loc[s].MW for s in self.seq2])
        self.qs, _ = get_qs(self.seq2,flexhis=True,pH=pH,residues=self.residues)
        self.alphas = self.lambdas*self.alpha

        self.calc_x_setup(comp_setup=comp_setup, d=0.58, n_per_res=2)
        self.init_bond_force()
        self.init_angle_force()

    def init_bond_force(self):
        self.bond_pairlist = []
        self.hb = interactions.init_bonded_interactions()
        self.basebase_pairlist = []
        self.scLJ_rna = interactions.init_scaled_LJ(self.eps_lj,self.rna_nb_cutoff)

    def init_angle_force(self):
        self.angle_list = []
        self.ha = interactions.init_angles()

    def get_forces(self):
        self.forces = [self.hb, self.scLJ_rna, self.ha]

    def calc_comp_seq(self):
        """ Calculate sequence of RNA. """

        records = read_fasta(self.ffasta)
        self.seq = str(records[self.name].seq) # one bead seq
        self.n_termini = [0]
        self.c_termini = [len(self.seq)-1]

        self.seq2 = '' # two bead seq
        for s in self.seq:
            self.seq2 += f'p{s}'

    @staticmethod
    def bond_check(i: int, j: int):
        """ Define bonded term conditions. """

        condition0 = (i%2 == 0) # phosphate
        condition1 = (j == i+2) # phosphate -- phosphate
        condition2 = (j == i+1) # phosphate -- base

        condition = condition0 and (condition1 or condition2)
        return condition

    @staticmethod
    def angle_check(i: int, j: int):
        """ Define angle term conditions. """

        condition = (i%2 == 0) and (j == i+4)
        return condition

    @staticmethod
    def basebase_check(i: int, j: int):
        """ Base-base interaction conditions. """

        condition = (i%2 == 1) and (j == i+2)
        return condition

    def calc_x_setup(self, comp_setup: str = 'spiral', d: float = 0.58, n_per_res: int = 2):
        if comp_setup == 'spiral':
            self.xinit = build.build_spiral(self.bondlengths[1::2], arc=d, n_per_res=n_per_res)
        else: # don't allow 'compact' setup for two-bead model
            z_bondlengths = self.bondlengths[::2]
            self.xinit = build.build_linear(z_bondlengths, n_per_res=n_per_res, ys=self.bondlengths)

    def add_bonds(self, offset):
        exclusion_map = []
        for i in range(0, self.nbeads-1):
            for j in range(i, self.nbeads):
                if self.bond_check(i,j): # p-p and p-b
                    d = self.bondlengths[j]
                    if j%2==0:
                        rna_kb = self.rna_kb1
                    else:
                        rna_kb = self.rna_kb2
                    bidx = self.hb.addBond(
                        i+offset, j+offset, d*unit.nanometer,
                        rna_kb*unit.kilojoules_per_mole/(unit.nanometer**2))
                    exclusion_map.append([i+offset,j+offset])
                    self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,d,rna_kb])
                if self.basebase_check(i,j): # restrain neighboring bases
                    sig = self.rna_nb_sigma
                    lam = (self.lambdas[i] + self.lambdas[j]) / 2.
                    n = self.rna_nb_scale
                    bidx = self.scLJ_rna.addBond(
                        i+offset,j+offset,
                        [sig*unit.nanometer, lam*unit.dimensionless, n*unit.dimensionless])
                    self.basebase_pairlist.append(
                        [i+offset+1,j+offset+1,bidx,sig,lam,n]
                    )
                    exclusion_map.append([i+offset,j+offset])
        return exclusion_map

    def add_angles(self, offset):
        exclusion_map = []
        for i in range(0, self.nbeads-1):
            for j in range(i, self.nbeads):
                if self.angle_check(i,j):
                    bidx = self.ha.addAngle(
                        i+offset,i+2+offset,i+4+offset,
                        self.rna_pa*unit.radian,
                        self.rna_ka*unit.kilojoules_per_mole/(unit.radian**2))
                    self.angle_list.append(
                        [i+offset+1,i+2+offset+1,i+4+offset+1,bidx,self.rna_pa,self.rna_ka]
                    )
                    exclusion_map.append([i+offset,j+offset])
        return exclusion_map

    def write_bonds(self, path):
        """ Write bonds and angles to file. """

        with open(f'{path}/bonds_{self.name}.txt','w') as f:
            f.write('i\tj\tb_idx\td[nm]\tk[kJ/mol/nm^2]\n')
            for b in self.bond_pairlist:
                f.write(f'{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{b[3]:.4f}\t{b[4]:.4f}\n')

        with open(f'{path}/basebase_{self.name}.txt','w') as f:
            f.write('i\tj\tb_idx\tsig\tlam\tn\n')
            for b in self.basebase_pairlist:
                f.write(f'{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{b[3]:.4f}\t{b[4]:.4f}\t{b[5]}\n')

        with open(f'{path}/angles_{self.name}.txt','w') as f:
            f.write('i\tj\tk\tb_idx\tsig\tlam\n')
            for b in self.angle_list:
                f.write(f'{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{int(b[3])}\t{b[4]:.4f}\t{b[5]:.4f}\n')

class Lipid(Component):
    """ Component lipid. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def calc_properties(self, pH: float = 7.0, verbose: bool = False, comp_setup: str = 'spiral'):
        """ Lipid properties. """

        super().calc_properties(pH=pH, verbose=verbose)
        self.calc_x_setup(comp_setup=comp_setup) # can be overwritten in custom component
        # self.calc_bondlength_map()

    @staticmethod
    def bond_check(i: int, j: int):
        """ Define bonded term conditions. """

        condition = (j == i+1) or (j == i+2)
        return condition

    def init_bond_force(self):
        self.bond_pairlist = []
        if self.molecule_type == 'lipid':
            self.hb = interactions.init_bonded_interactions()
            self.ha = interactions.init_angles()
        elif self.molecule_type == 'cooke_lipid':
            self.wcafene = interactions.init_wcafene(self.eps_lj)
            self.hb = interactions.init_bonded_interactions()

    def add_bonds(self, offset):
        """ Lipid bonds """
        exclusion_map = [] # for ah, yu etc.
        for i in range(0,self.nbeads-1):
            for j in range(i, self.nbeads):
                if self.bond_check(i,j):
                    d = self.calc_bondlength(i,j)
                    if j-i == 1:
                        if self.molecule_type == 'cooke_lipid':
                            kfene = 30*3*self.eps_lj/d/d
                            bidx = self.wcafene.addBond(
                                i+offset, j+offset,
                                [d*unit.nanometer, kfene*unit.kilojoules_per_mole/(unit.nanometer**2)])
                            self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,d,kfene]) # 1-based
                        elif self.molecule_type == 'lipid':
                            bidx = self.hb.addBond(
                                i+offset, j+offset, d*unit.nanometer,
                                1700*unit.kilojoules_per_mole/(unit.nanometer**2))
                            self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,d,1700]) # 1-based
                        exclusion_map.append([i+offset,j+offset])
                    else:
                        if self.molecule_type == 'cooke_lipid':
                            kbend = 30*self.eps_lj/d/d
                            bidx = self.hb.addBond(
                                i+offset, j+offset,
                                4*d*unit.nanometer, kbend*unit.kilojoules_per_mole/(unit.nanometer**2))
                            self.bond_pairlist.append([i+offset+1,j+offset+1,bidx,4*d,kbend]) # 1-based
                        else:
                            angle = 2/3*np.pi if i == 0 else np.pi
                            k_angle = 7/2 if i == 0 else 7
                            self.ha.addAngle(
                                i+offset, i+offset+1, j+offset,
                                angle*unit.radian, k_angle*unit.kilojoules_per_mole/(unit.radian**2))
        return exclusion_map

class Crowder(Component):
    """ Component Crowder. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def calc_properties(self, pH: float = 7.0, verbose: bool = False, comp_setup: str = 'spiral'):
        """ Crowder properties. """

        super().calc_properties(pH=pH, verbose=verbose)
        self.calc_x_setup(comp_setup=comp_setup) # can be overwritten in custom component

    @staticmethod
    def bond_check(i: int, j: int):
        """ Define bonded term conditions. """

        condition = (j == i+1)
        return condition

class Cyclic(Protein):
    """ Cyclic peptide. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def bond_check(self, i: int, j: int):
        """ Define bonded term conditions. """

        condition0 = (j == i+1)
        condition1 = ((j == self.nbeads - 1) and i == 0)
        condition = condition0 or condition1
        return condition

class Seastar(Protein):
    """ Branched peptide. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def bond_check(self, i: int, j: int):
        """ Define bonded term conditions. """

        if self.n_ends in [0,1,2]:
            return super().bond_check(i,j)
        else:
            if (self.nbeads-1) % self.n_ends == 0:
                branch_length = int((self.nbeads-1) / self.n_ends)
            else:
                branch_length = int((self.nbeads-1) / self.n_ends) + 1

            condition0 = (j == i+1) and ((j-1) % branch_length != 0)
            condition1 = (i == 0) and ((j-1) % branch_length == 0)

            condition = condition0 or condition1
            return condition

class PTMProtein(Protein):
    """ Branched peptide. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def calc_comp_seq(self):
        """ Calculate sequence of Protein + PTM. """

        records = read_fasta(self.ffasta)
        self.seq = str(records[self.name].seq) # one bead seq
        self.nbeads_protein = len(self.seq)
        self.ptm_seq = str(records[self.ptm_name].seq)

        for ptm_idx in self.ptm_locations: # 1-based
            self.seq = self.seq + self.ptm_seq

        self.n_termini = [0]
        self.c_termini = [len(self.seq)-1]

    def bond_check(self, i: int, j: int):
        """ Define bonded term conditions. """

        # residue-residue bond (protein)
        if (i < self.nbeads_protein - 1) and (j == i+1):
            return True
        
        ptm_seqlocs = []
        # residue-PTM bond
        for idx, ptm_loc in enumerate(self.ptm_locations):
            ptm_seqloc = self.nbeads_protein + idx * len(self.ptm_seq) # position of connecting PTM bead in sequence
            ptm_seqlocs.append(ptm_seqloc)
            if (i == ptm_loc - 1) and (j == ptm_seqloc):
                return True
        
        # PTM-PTM bond
        if i >= self.nbeads_protein:
            if (j == i+1) and (j not in ptm_seqlocs): # avoid bonding different PTMs
                return True
        return False
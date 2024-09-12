from .sequence import seq_from_pdb, read_fasta, get_qs, patch_terminal_qs
from .analysis import self_distances
from calvados import build
from pandas.core.frame import DataFrame

import numpy as np

class Component:
    """ Generic component of the system. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        self.name = name
        self.molecule_type = comp_dict.get('molecule_type', defaults.get('molecule_type','protein'))
        self.nmol = comp_dict.get('nmol', defaults.get('nmol',1))
        self.restr = comp_dict.get('restraint', defaults.get('restraint',False))
        self.charge_termini = comp_dict.get('charge_termini', defaults.get('charge_termini','both'))
        self.alpha = comp_dict.get('alpha', defaults.get('alpha',0))

    def calc_comp_seq(self, ffasta: str = None, pdb_folder: str = None):
        """ Calculate sequence of component. """

        if self.restr:
            self.seq, self.n_termini, self.c_termini = seq_from_pdb(f'{pdb_folder}/{self.name}.pdb')
        else:
            records = read_fasta(ffasta)
            self.seq = str(records[self.name].seq)
            self.n_termini = [0]
            self.c_termini = [len(self.seq)-1]

    def calc_comp_properties(self, residues: DataFrame, pH: float = 7.0, verbose: bool = False):
        """ Calculate component properties (sigmas, lambdas, qs etc.) """

        self.nres = len(self.seq)
        self.sigmas = np.array([residues.loc[s].sigmas for s in self.seq])
        self.lambdas = np.array([residues.loc[s].lambdas for s in self.seq])
        self.bondlengths = np.array([residues.loc[s].bondlength for s in self.seq])
        self.mws = np.array([residues.loc[s].MW for s in self.seq])
        self.qs, _ = get_qs(self.seq,flexhis=True,pH=pH,residues=residues)
        self.alphas = self.lambdas*self.alpha

    def calc_dmap(self):
        self.dmap = self_distances(self.xinit)

    def calc_x_linear(self, d: float = 0.38, spiral: bool = False):
        if spiral:
            self.xinit = build.build_spiral(self.nres, arc=d)
        else:
            self.xinit = build.build_linear(self.bondlengths)

    def calc_bond_map(self):
        self.bond_map = np.zeros((self.nres, self.nres), dtype=bool)
        for idx in range(self.nres):
            for jdx in range(self.nres):
                self.bond_map[idx,jdx] = self.bond_check(idx,jdx)

    @staticmethod
    def bond_check(i: int, j: int):
        """ Placeholder for molecule-specific method. """
        return False

class Protein(Component):
    """ Component protein. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def calc_x_from_pdb(self, pdb_folder: str, use_com: bool = True):
        """ Calculate protein positions from pdb. """

        input_pdb = f'{pdb_folder}/{self.name}.pdb'
        self.xinit = build.geometry_from_pdb(input_pdb,use_com=use_com) # read from pdb

    def calc_ssdomains(self, fdomains):
        """ Get bounds for restraints (harmonic). """

        self.ssdomains = build.get_ssdomains(self.name,fdomains)
        # print(f'Setting {self.restraint_type} restraints for {comp.name}')

    def calc_go_scale(self,
            pdb_folder: str, bfac_width: float = 50., bfac_shift: float = 0.8,
            pae_width: float = 8., pae_shift: float = 0.4, colabfold: int = 0):
        """ Calculate scaling of Go potential for all residue pairs. """

        input_pdb = f'{pdb_folder}/{self.name}.pdb'
        bfac = build.bfac_from_pdb(input_pdb,confidence=0.)
        bfac_map = np.add.outer(bfac,bfac) / 2.
        bfac_sigm = np.exp(bfac_width*(bfac_map-bfac_shift)) / (np.exp(bfac_width*(bfac_map-bfac_shift)) + 1)

        input_pae = f'{pdb_folder}/{self.name}.json'
        pae = build.load_pae(input_pae,symmetrize=True,colabfold=colabfold) / 10. # in nm
        pae_sigm = np.exp(-pae_width*(pae-pae_shift)) / (np.exp(-pae_width*(pae-pae_shift)) + 1)
        # pae_inv = build.load_pae_inv(input_pae,colabfold=self.colabfold)
        scaled_LJYU_pairlist = []
        self.scale = bfac_sigm * pae_sigm # restraint scale
        self.bondscale = np.maximum(0, 1. - 5.* self.scale) # residual interactions for low restraints

    def calc_comp_properties(self, residues: DataFrame, pH: float = 7.0, verbose: bool = False):
        """ Protein properties. """

        super().calc_comp_properties(residues=residues, pH=pH, verbose=verbose)
        # fix mass of termini
        self.mws[self.n_termini] += 2
        self.mws[self.c_termini] += 16
        # fix charge of termini
        if verbose:
            print(f'Adding charges for {self.charge_termini} termini of {self.name}.', flush=True)
        self.qs = patch_terminal_qs(self.qs,self.n_termini,self.c_termini,loc=self.charge_termini)

        super().calc_bond_map()
        self.calc_bondlength_map()

    def calc_bondlength_map(self):
        """ Map of bondlength for all i,j. """

        self.bondlength_map = np.zeros((self.nres, self.nres))

    def bond_check(self, i: int, j: int):
        """ Define bonded term conditions. """

        condition = (j == i+1)
        condition_termini = (i not in self.c_termini) and (j not in self.n_termini)
        return condition and condition_termini

class RNA(Component):
    """ Component RNA. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    @staticmethod
    def bond_check(i: int, j: int):
        """ Define bonded term conditions. """

        condition0 = (j == i+2) and (i % 2 == 0) # phosphate -- phosphate
        condition1 = (j == i+1) and (i % 2 == 0) # phosphate -- base

        condition = condition0 or condition1
        return condition

class Lipid(Component):
    """ Component lipid. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def calc_comp_properties(self, residues: DataFrame, pH: float = 7.0, verbose: bool = False):
        """ Lipid properties. """

        super().calc_comp_properties(residues=residues, pH=pH, verbose=verbose)
        super().calc_bond_map()

    @staticmethod
    def bond_check(i: int, j: int):
        """ Define bonded term conditions. """

        condition = (j == i+1) or (j == i+2)
        return condition

class Crowder(Component):
    """ Component crowder. """

    def __init__(self, name: str, comp_dict: dict, defaults: dict):
        super().__init__(name, comp_dict, defaults)

    def calc_comp_properties(self, residues: DataFrame, pH: float = 7.0, verbose: bool = False):
        """ Crowder properties. """

        super().calc_comp_properties(residues=residues, pH=pH, verbose=verbose)
        super().calc_bond_map()

    @staticmethod
    def bond_check(i: int, j: int):
        """ Define bonded term conditions. """

        condition = (j == i+1)
        return condition

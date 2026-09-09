import os
from os import PathLike

import numpy as np
from numpy.typing import NDArray
from openmm import unit
from pandas import read_csv
from scipy.special import expit

from calvados import build, interactions

from .analysis import self_distances
from .sequence import (
    get_qs,
    patch_terminal_mws,
    patch_terminal_qs,
    read_fasta,
    seq_from_pdb,
)
from .inputmodels import ComponentInput


class Component:
    """Represent a coarse-grained molecular component.

    This base class provides shared sequence, property, coordinate, and bond setup.
    """

    def __init__(
            self,
            name: str,
            params: ComponentInput,
        ):
        """Initialize a component from explicit and default properties."""
        self.name = name

        # read component properties
        for key, val in params.model_dump().items():
            setattr(self, key, val)

        # read residue parameters from file
        try:
            self.residues = read_csv(self.fresidues).set_index("one")
        except AttributeError:
            raise FileNotFoundError(
                f"Residue parameter file name (fresidues) not supplied to component {name}."
            )
        self.comp_setup = "compact"

    def calc_comp_seq(self) -> None:
        """Calculate the component sequence."""

        if self.restraint:
            pdb_file = f"{self.pdb_folder}/{self.name}.pdb"
            cif_file = f"{self.pdb_folder}/{self.name}.cif"
            if os.path.isfile(cif_file):
                self.seq, self.n_termini, self.c_termini = seq_from_pdb(cif_file)
            else:
                self.seq, self.n_termini, self.c_termini = seq_from_pdb(pdb_file)
        else:
            records = read_fasta(self.ffasta)
            self.seq = str(records[self.name].seq)
            self.n_termini = [0]
            self.c_termini = [len(self.seq) - 1]

    def calc_properties(self, pH: float = 7.0, verbose: bool = False) -> None:
        """Calculate bead properties for the component."""

        self.calc_comp_seq()
        self.nres = len(self.seq)
        self.nbeads = self.nres
        self.sigmas = np.array([self.residues.loc[s].sigmas for s in self.seq])
        self.lambdas = np.array([self.residues.loc[s].lambdas for s in self.seq])
        self.bondlengths = np.array([self.residues.loc[s].bondlength for s in self.seq])
        self.mws = np.array([self.residues.loc[s].MW for s in self.seq])
        self.qs, _ = get_qs(self.seq, flexhis=True, pH=pH, residues=self.residues)
        self.alphas = self.lambdas * self.alpha
        self.init_bond_force()

    def calc_dmap(self) -> None:
        """Calculate the intracomponent distance map."""
        if self.periodic:
            self.dmap = self_distances(self.xinit, self.dimensions)
        else:
            self.dmap = self_distances(self.xinit)

    def calc_x_setup(
        self,
        d: float = 0.38,
        n_per_res: int = 1,
        ys: NDArray[np.float64] | None = None,
    ) -> None:
        """Generate initial coordinates using the requested arrangement."""
        if self.comp_setup == "spiral":
            self.xinit = build.build_spiral(
                self.bondlengths, arc=d, n_per_res=n_per_res
            )
        elif self.comp_setup == "compact":
            self.xinit = build.build_compact(self.nbeads, d=d)
        else:
            self.xinit = build.build_linear(
                self.bondlengths, n_per_res=n_per_res, ys=ys
            )

    def bond_check(self, i: int, j: int) -> bool:
        """Return whether two beads should be bonded."""
        return False

    def calc_bondlength(self, i: int, j: int) -> float:
        """Calculate the equilibrium bond length between two beads."""
        d0 = 0.5 * (self.bondlengths[i] + self.bondlengths[j])
        return d0

    def init_bond_force(self) -> None:
        """Initialize the harmonic bond force and its bond records."""
        self.bond_pairlist = []
        self.hb = interactions.init_bonded_interactions()

    def add_bonds(self, offset: int) -> list[list[int]]:
        """Add component bonds and return their nonbonded exclusions."""
        exclusion_map = []  # for ah, yu etc.
        for i in range(0, self.nbeads - 1):
            for j in range(i, self.nbeads):
                if self.bond_check(i, j):
                    d = self.calc_bondlength(i, j)
                    bidx = self.hb.addBond(
                        i + offset,
                        j + offset,
                        d * unit.nanometer,
                        self.kb * unit.kilojoules_per_mole / (unit.nanometer**2),
                    )
                    self.bond_pairlist.append(
                        [i + offset + 1, j + offset + 1, bidx, d, self.kb]
                    )  # 1-based
                    exclusion_map.append([i + offset, j + offset])
        return exclusion_map

    def get_forces(self) -> None:
        """Collect the component forces that belong in the system."""
        self.forces = [self.hb]

    def write_bonds(self, path: str | PathLike[str]) -> None:
        """Write bond records to a file."""

        with open(f"{path}/bonds_{self.name}.txt", "w") as f:
            f.write("i\tj\tb_idx\td[nm]\tk[kJ/mol/nm^2]\n")
            for b in self.bond_pairlist:
                f.write(
                    f"{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{b[3]:.4f}\t{b[4]:.4f}\n"
                )


class Protein(Component):
    """Represent a one-bead-per-residue protein component.

    Proteins may use harmonic or Go-like restraints derived from a structure.
    """

    @staticmethod
    def get_input_structure_file(pdb_folder: str | PathLike[str], name: str) -> str:
        """Return the available CIF or PDB structure filename."""
        pdb_file = f"{pdb_folder}/{name}.pdb"
        cif_file = f"{pdb_folder}/{name}.cif"

        if os.path.isfile(cif_file):
            return cif_file
        elif os.path.isfile(pdb_file):
            return pdb_file
        else:
            raise ValueError("Input structure file must be of type pdb or cif")

    def calc_x_from_pdb(self) -> None:
        """Load protein coordinates from a PDB or CIF structure."""

        structure_file = self.get_input_structure_file(self.pdb_folder, self.name)
        self.xinit, self.dimensions = build.geometry_from_pdb(
            structure_file,
            use_com=self.use_com,
        )  # read from pdb

    def calc_ssdomains(self) -> None:
        """Load the structured domains used for harmonic restraints."""

        self.ssdomains = build.get_ssdomains(self.name, self.fdomains)

    def calc_go_scale(
        self, bscale_shift: float = 0.1, bscale_width: float = 80
    ) -> None:
        """Calculate Go-potential scaling for all residue pairs."""

        structure_file = self.get_input_structure_file(self.pdb_folder, self.name)
        bfac = build.bfac_from_pdb(structure_file, confidence=0.0)
        self.bfac_map = np.minimum.outer(bfac, bfac)
        bfac_sigm = expit(self.bfac_width * (self.bfac_map - self.bfac_shift))

        pae_file = f"{self.pdb_folder}/{self.name}.json"
        self.pae = (
            build.load_pae(pae_file, symmetrize=True, colabfold=self.colabfold) / 10.0
        )  # in nm
        pae_sigm = expit(-self.pae_width * (self.pae - self.pae_shift))

        self.scale = bfac_sigm * pae_sigm  # restraint scale
        self.bondscale = expit(-bscale_width * (self.scale - bscale_shift))

    def calc_properties(
        self, pH: float = 7.0, verbose: bool = False,
    ) -> None:
        """Calculate protein properties and initial coordinates."""

        super().calc_properties(pH=pH, verbose=verbose)

        # fix charge and mw of termini
        if verbose:
            print(
                f"Adding charges for {self.charge_termini} termini of {self.name}.",
                flush=True,
            )
        self.qs = patch_terminal_qs(
            self.qs, self.n_termini, self.c_termini, loc=self.charge_termini
        )
        self.mws = patch_terminal_mws(
            self.mws, self.n_termini, self.c_termini, loc=self.charge_termini
        )

        if self.restraint:
            self.calc_x_from_pdb()
            self.calc_dmap()
            if self.restraint_type == "harmonic":
                self.calc_ssdomains()
            elif self.restraint_type == "go":
                self.calc_go_scale()
        else:
            self.calc_x_setup()

    def calc_bondlength(
        self,
        i: int,
        j: int,
        min_scale: float = 0.05,
        cutoff_mix_in_LJYU: float = 0.15,
    ) -> float:
        """Calculate a protein bond length, including structural restraints."""
        d0 = 0.5 * (self.bondlengths[i] + self.bondlengths[j])
        if self.restraint:
            if self.restraint_type == "harmonic":
                ss = build.check_ssdomain(self.ssdomains, i, j, req_both=False)
                d = self.dmap[i, j] if ss else d0
            elif self.restraint_type == "go":
                if self.scale[i, j] < min_scale:
                    d = d0
                elif self.scale[i, j] > cutoff_mix_in_LJYU:
                    d = self.dmap[i, j]
                else:
                    d = (
                        self.bondscale[i, j] * d0
                        + (1.0 - self.bondscale[i, j]) * self.dmap[i, j]
                    )
            else:
                raise ValueError("Restraint type must be harmonic or go.")
        else:
            d = d0
        return d

    def bond_check(self, i: int, j: int) -> bool:
        """Return whether two protein beads should be bonded."""

        condition = j == i + 1
        condition_termini = (i not in self.c_termini) and (j not in self.n_termini)
        return condition and condition_termini

    def init_restraint_force(
        self, eps_lj=None, cutoff_lj=None, cutoff_yu=None, eps_yu=None, k_yu=None
    ) -> None:
        """Initialize protein restraint forces and their pair records."""
        if self.restraint_type not in ["harmonic", "go"]:
            raise ValueError("Protein restraint type must be harmonic or go.")

        self.cs = interactions.init_restraints(self.restraint_type)
        self.restr_pairlist = []
        if self.restraint_type == "go":
            self.scLJ_pairlist = []
            self.scYU_pairlist = []
            self.scLJ = interactions.init_scaled_LJ(eps_lj, cutoff_lj)
            self.scYU = interactions.init_scaled_YU(eps_yu, k_yu, cutoff_yu)

    def add_restraints(
        self,
        offset: int,
        min_scale: float = 0.05,
        cutoff_mix_in_LJYU: float = 0.15,
    ) -> list[list[int]]:
        """Add protein restraints and return their nonbonded exclusions."""
        exclusion_map = []  # for ah, yu etc.

        for i in range(0, self.nbeads - 2):
            for j in range(i + 2, self.nbeads):
                # check if below cutoff
                if self.dmap[i, j] > self.cutoff_restr:
                    continue
                # harmonic
                if self.restraint_type == "harmonic":
                    ss = build.check_ssdomain(self.ssdomains, i, j, req_both=True)
                    if not ss:
                        continue
                    k = self.k_harmonic
                # go
                elif self.restraint_type == "go":
                    if self.scale[i, j] < min_scale:
                        continue
                    k = self.k_go * self.scale[i, j]
                    # add scaled pseudo LJ, YU for low restraints
                    if self.scale[i, j] < cutoff_mix_in_LJYU:  # but >= min_scale
                        self.scLJ, scaled_pair = interactions.add_scaled_lj(
                            self.scLJ, i, j, offset, self
                        )
                        self.scLJ_pairlist.append(scaled_pair)
                        if self.qs[i] * self.qs[j] != 0.0:
                            self.scYU, scaled_pair = interactions.add_scaled_yu(
                                self.scYU, i, j, offset, self
                            )
                            self.scYU_pairlist.append(scaled_pair)

                self.cs, restr_pair = interactions.add_single_restraint(
                    self.cs,
                    self.restraint_type,
                    self.dmap[i, j],
                    k,
                    i + offset,
                    j + offset,
                )
                self.restr_pairlist.append(restr_pair)
                exclusion_map.append([i + offset, j + offset])
        return exclusion_map

    def write_restraints(self, path: str | PathLike[str]) -> None:
        """Write protein restraint records to files."""

        with open(f"{path}/restr_{self.name}.txt", "w") as f:
            f.write("i j d[nm] fc\n")
            for r in self.restr_pairlist:
                f.write(f"{int(r[0])} {int(r[1])} {r[2]:.4f} {r[3]:.4f}\n")

        if self.restraint_type == "go":
            with open(f"{path}/scaled_LJ_{self.name}.txt", "w") as f:
                f.write(
                    "i+offset+1, j+offset+1, s, l, comp.bondscale[i,j]\n"
                )  # 1-based
                for r in self.scLJ_pairlist:
                    f.write(
                        f"{int(r[0])} {int(r[1])} {r[2]:.4f} {r[3]:.4f} {r[4]:.4f}\n"
                    )

            with open(f"{path}/scaled_YU_{self.name}.txt", "w") as f:
                f.write("i+offset+1, j+offset+1, comp.bondscale[i,j]\n")  # 1-based
                for r in self.scYU_pairlist:
                    f.write(f"{int(r[0])} {int(r[1])} {r[2]:.4f}\n")

    def get_forces(self) -> None:
        """Collect protein bond and restraint forces for the system."""
        self.forces = [self.hb]
        if self.restraint:
            self.forces.append(self.cs)
            if self.restraint_type == "go":
                self.forces.extend([self.scLJ, self.scYU])


class RNA(Component):
    """Represent an RNA component with phosphate and nucleotide beads.

    RNA components include backbone bonds, angles, and neighboring-base forces.
    """

    def __init__(self,
            name: str,
            params: ComponentInput,
    ) -> None:
        super().__init__(name, params)
        self.comp_setup = "spiral"

    def calc_x_from_pdb(self) -> None:
        """Calculate RNA positions from a PDB structure."""
        pdb_file = f"{self.pdb_folder}/{self.name}.pdb"
        self.xinit, self.dimensions = build.geometry_from_pdb_rna(
            pdb_file, use_com=self.use_com
        )  # read from pdb

    def calc_ssdomains(self) -> None:
        """Map harmonic-restraint domains from residues to RNA beads."""
        seq_ssdomains = build.get_ssdomains(self.name, self.fdomains)
        ssdomains_bead = []
        for seq_ssdomain in seq_ssdomains:
            ss_domain_bead = []
            for s in seq_ssdomain:
                ss_domain_bead.append(2 * s)
                ss_domain_bead.append(2 * s + 1)
            ssdomains_bead.append(ss_domain_bead)
        self.ssdomains = ssdomains_bead

    def calc_properties(
        self, pH: float = 7.0, verbose: bool = False,
    ) -> None:
        """Calculate RNA properties and initial coordinates."""

        self.calc_comp_seq()  # --> seq and seq2
        self.nres = len(self.seq)
        self.nbeads = len(self.seq2)

        self.sigmas = np.array([self.residues.loc[s].sigmas for s in self.seq2])
        self.lambdas = np.array([self.residues.loc[s].lambdas for s in self.seq2])
        self.bondlengths = np.array(
            [self.residues.loc[s].bondlength for s in self.seq2]
        )
        self.mws = np.array([self.residues.loc[s].MW for s in self.seq2])
        self.qs, _ = get_qs(self.seq2, flexhis=True, pH=pH, residues=self.residues)
        self.alphas = self.lambdas * self.alpha

        if self.restraint:
            self.calc_x_from_pdb()
            self.calc_dmap()
            self.calc_angmap()
            if self.restraint_type == "harmonic":
                self.calc_ssdomains()
        else:
            self.calc_x_setup(d=0.59, n_per_res=2)

        self.init_bond_force()
        self.init_angle_force()

    def init_bond_force(self) -> None:
        """Initialize RNA bond and neighboring-base forces."""
        self.bond_pairlist = []
        self.hb = interactions.init_bonded_interactions()
        self.basebase_pairlist = []
        self.scLJ_rna = interactions.init_scaled_LJ(self.eps_lj, self.rna_nb_cutoff)

    def init_angle_force(self) -> None:
        """Initialize the RNA angle force and its angle records."""
        self.angle_list = []
        self.ha = interactions.init_angles()

    def get_forces(self) -> None:
        """Collect RNA bond, angle, base, and restraint forces."""
        self.forces = [self.hb, self.scLJ_rna, self.ha]
        if self.restraint:
            self.forces.append(self.cs)

    def calc_comp_seq(self) -> None:
        """Calculate the one- and two-bead RNA sequences."""

        if self.restraint:
            four_type_seq, n_termini_seq, c_termini_seq = seq_from_pdb(
                f"{self.pdb_folder}/{self.name}.pdb"
            )
            seq_ssdomains = build.get_ssdomains(self.name, self.fdomains)
            seq = []
            for i in range(len(four_type_seq)):
                ss_residue_condition = any(
                    i in seq_ssdomain for seq_ssdomain in seq_ssdomains
                )
                seq.append("s" if ss_residue_condition else "r")
            self.seq = "".join(seq)
        else:
            records = read_fasta(self.ffasta)
            self.seq = str(records[self.name].seq)  # one bead seq
            n_termini_seq = [0]
            c_termini_seq = [len(self.seq) - 1]

        self.seq2 = "".join(f"p{s}" for s in self.seq)  # two bead seq

        self.n_termini = [x for i in n_termini_seq for x in (2 * i, 2 * i + 1)]
        self.c_termini = [x for i in c_termini_seq for x in (2 * i, 2 * i + 1)]

    def calc_bondlength(self, i: int, j: int) -> float:
        """Calculate an RNA bond length, including structural restraints."""
        d0 = self.bondlengths[j]
        if self.restraint:
            if self.restraint_type == "harmonic":
                ss = build.check_ssdomain(self.ssdomains, i, j, req_both=False)
                d = self.dmap[i, j] if ss else d0
            else:
                raise ValueError("Restraint type must be harmonic.")
        else:
            d = d0
        return d

    def calc_rna_nb_sigma_length(self, i: int, j: int) -> float:
        """Calculate the sigma value for neighboring RNA bases."""
        sig0 = self.rna_nb_sigma
        if self.restraint:
            if self.restraint_type == "harmonic":
                ss = build.check_ssdomain(self.ssdomains, i, j, req_both=False)
                sig = self.dmap[i, j] / (2 ** (1 / 6)) if ss else sig0
            else:
                raise ValueError("Restraint type must be harmonic.")
        else:
            sig = sig0
        return sig

    def calc_angmap(self) -> None:
        """Calculate equilibrium backbone angles from initial coordinates."""
        nbeads = len(self.xinit)
        angmap = np.zeros(nbeads)
        pos = self.xinit
        for i in range(0, nbeads - 4, 2):
            v1 = pos[i] - pos[i + 2]
            v2 = pos[i + 4] - pos[i + 2]
            cos = np.dot(v1, v2) / np.linalg.norm(v1) / np.linalg.norm(v2)
            cos = np.clip(cos, -1.0, 1.0)
            angmap[i] = np.arccos(cos)
        self.angmap = angmap

    def calc_angle(self, i: int, j: int) -> float:
        """Calculate an RNA backbone angle, including structural restraints."""
        ang0 = self.rna_pa
        if self.restraint:
            if self.restraint_type == "harmonic":
                ss = build.check_ssdomain(self.ssdomains, i, j, req_both=False)
                ang = self.angmap[i] if ss else ang0
            else:
                raise ValueError("Restraint type must be harmonic.")
        else:
            ang = ang0
        return ang

    def bond_check(self, i: int, j: int) -> bool:
        """Return whether two RNA beads should be bonded."""

        condition0 = i % 2 == 0  # phosphate
        condition1 = j == i + 2  # phosphate -- phosphate
        condition2 = j == i + 1  # phosphate -- base

        condition = condition0 and (condition1 or condition2)

        condition_termini = not ((i in self.c_termini) and (j in self.n_termini))
        return condition and condition_termini

    def angle_check(self, i: int, j: int) -> bool:
        """Return whether two RNA beads define a backbone angle."""

        condition = (i % 2 == 0) and (j == i + 4)
        condition_termini = (i + 2 not in self.n_termini) and (
            i + 2 not in self.c_termini
        )
        return condition and condition_termini

    def basebase_check(self, i: int, j: int) -> bool:
        """Return whether two RNA beads are neighboring bases."""

        condition = (i % 2 == 1) and (j == i + 2)
        condition_termini = not ((i in self.c_termini) and (j in self.n_termini))
        return condition and condition_termini

    def calc_x_setup(
        self, d: float = 0.59, n_per_res: int = 2, ys: NDArray[np.float64] | None = None,
    ) -> None:
        """Generate initial coordinates for the two-bead RNA model."""
        if self.comp_setup == "spiral":
            self.xinit = build.build_spiral(
                self.bondlengths[1::2], arc=d, n_per_res=n_per_res
            )
        else:  # don't allow 'compact' setup for two-bead model
            z_bondlengths = self.bondlengths[::2]
            self.xinit = build.build_linear(
                z_bondlengths, n_per_res=n_per_res, ys=self.bondlengths
            )

    def add_bonds(self, offset: int) -> list[list[int]]:
        """Add RNA bonds and neighboring-base forces and return exclusions."""
        exclusion_map = []
        for i in range(0, self.nbeads - 1):
            for j in range(i, self.nbeads):
                if self.bond_check(i, j):  # p-p and p-b
                    d = self.calc_bondlength(i, j)
                    if j % 2 == 0:
                        rna_kb = self.rna_kb1
                    else:
                        rna_kb = self.rna_kb2
                    bidx = self.hb.addBond(
                        i + offset,
                        j + offset,
                        d * unit.nanometer,
                        rna_kb * unit.kilojoules_per_mole / (unit.nanometer**2),
                    )
                    exclusion_map.append([i + offset, j + offset])
                    self.bond_pairlist.append(
                        [i + offset + 1, j + offset + 1, bidx, d, rna_kb]
                    )
                if self.basebase_check(i, j):  # restrain neighboring bases
                    sig = self.calc_rna_nb_sigma_length(i, j)
                    lam = (self.lambdas[i] + self.lambdas[j]) / 2.0
                    n = self.rna_nb_scale
                    bidx = self.scLJ_rna.addBond(
                        i + offset,
                        j + offset,
                        [
                            sig * unit.nanometer,
                            lam * unit.dimensionless,
                            n * unit.dimensionless,
                        ],
                    )
                    self.basebase_pairlist.append(
                        [i + offset + 1, j + offset + 1, bidx, sig, lam, n]
                    )
                    exclusion_map.append([i + offset, j + offset])
        return exclusion_map

    def add_angles(self, offset: int) -> list[list[int]]:
        """Add RNA backbone angles and return their nonbonded exclusions."""
        exclusion_map = []
        for i in range(0, self.nbeads - 1):
            for j in range(i, self.nbeads):
                if self.angle_check(i, j):
                    rna_pa = self.calc_angle(i, j)
                    bidx = self.ha.addAngle(
                        i + offset,
                        i + 2 + offset,
                        i + 4 + offset,
                        rna_pa * unit.radian,
                        self.rna_ka * unit.kilojoules_per_mole / (unit.radian**2),
                    )
                    self.angle_list.append(
                        [
                            i + offset + 1,
                            i + 2 + offset + 1,
                            i + 4 + offset + 1,
                            bidx,
                            rna_pa,
                            self.rna_ka,
                        ]
                    )
                    exclusion_map.append([i + offset, j + offset])
        return exclusion_map

    def init_restraint_force(
        self, eps_lj=None, cutoff_lj=None, eps_yu=None, k_yu=None
    ) -> None:
        """Initialize the RNA restraint force and its pair records."""
        self.cs = interactions.init_restraints(self.restraint_type)
        self.restr_pairlist = []

    def restraint_check(self, i: int, j: int) -> bool:
        """Return whether a bead pair is eligible for a restraint."""

        bond_condition = self.bond_check(i, j)
        angle_condition = self.angle_check(i, j)
        basebase_condition = self.basebase_check(i, j)
        condition = (
            (not bond_condition) and (not angle_condition) and (not basebase_condition)
        )

        condition_termini = (i in self.c_termini) or (j in self.n_termini)

        return condition or condition_termini

    def add_restraints(self, offset: int, min_scale: float = 0.1) -> list[list[int]]:
        """Add RNA restraints and return their nonbonded exclusions."""
        exclusion_map = []  # for ah, yu etc.
        for i in range(self.nbeads - 2):
            for j in range(i + 2, self.nbeads):
                if self.restraint_check(i, j):
                    # check if below cutoff
                    if self.dmap[i, j] > self.cutoff_restr:
                        continue
                    # harmonic
                    if self.restraint_type == "harmonic":
                        ss = build.check_ssdomain(self.ssdomains, i, j, req_both=True)
                        if not ss:
                            continue
                        k = self.k_harmonic
                        self.cs, restr_pair = interactions.add_single_restraint(
                            self.cs,
                            self.restraint_type,
                            self.dmap[i, j],
                            k,
                            i + offset,
                            j + offset,
                        )
                        self.restr_pairlist.append(restr_pair)
                        exclusion_map.append([i + offset, j + offset])
                    else:
                        raise ValueError("RNA restraint type must be harmonic.")
        return exclusion_map

    def write_bonds(self, path: str | PathLike[str]) -> None:
        """Write RNA bond, neighboring-base, and angle records to files."""

        with open(f"{path}/bonds_{self.name}.txt", "w") as f:
            f.write("i\tj\tb_idx\td[nm]\tk[kJ/mol/nm^2]\n")
            for b in self.bond_pairlist:
                f.write(
                    f"{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{b[3]:.4f}\t{b[4]:.4f}\n"
                )

        with open(f"{path}/basebase_{self.name}.txt", "w") as f:
            f.write("i\tj\tb_idx\tsig\tlam\tn\n")
            for b in self.basebase_pairlist:
                f.write(
                    f"{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{b[3]:.4f}\t{b[4]:.4f}\t{b[5]}\n"
                )

        with open(f"{path}/angles_{self.name}.txt", "w") as f:
            f.write("i\tj\tk\ta_idx\ta[rad]\tk[kJ/mol/rad^2]\n")
            for b in self.angle_list:
                f.write(
                    f"{int(b[0])}\t{int(b[1])}\t{int(b[2])}\t{int(b[3])}\t{b[4]:.4f}\t{b[5]:.4f}\n"
                )

    def write_restraints(self, path: str | PathLike[str]) -> None:
        """Write RNA restraint records to a file."""

        with open(f"{path}/restr_{self.name}.txt", "w") as f:
            f.write("i j d[nm] fc\n")
            for r in self.restr_pairlist:
                f.write(f"{int(r[0])} {int(r[1])} {r[2]:.4f} {r[3]:.4f}\n")


class Lipid(Component):
    """Represent a three-bead lipid component.

    Both harmonic-angle and Cooke-style lipid force variants are supported.
    """

    def __init__(self,
            name: str,
            params: ComponentInput,
    ) -> None:
        super().__init__(name, params)
        self.comp_setup = "linear"

    def calc_properties(
        self, pH: float = 7.0, verbose: bool = False,
    ):
        """Calculate lipid properties and initial coordinates."""

        super().calc_properties(pH=pH, verbose=verbose)
        self.calc_x_setup()  # can be overwritten in custom component

    @staticmethod
    def bond_check(i: int, j: int):
        """Return whether two lipid beads share a bond or angle."""

        condition = (j == i + 1) or (j == i + 2)
        return condition

    def init_bond_force(self):
        """Initialize the forces required by the selected lipid model."""
        self.bond_pairlist = []
        if self.molecule_type == "lipid":
            self.hb = interactions.init_bonded_interactions()
            self.ha = interactions.init_angles()
        elif self.molecule_type == "cooke_lipid":
            self.hb = interactions.init_bonded_interactions()
            self.wcafene = interactions.init_wcafene(self.eps_lj)

    def add_bonds(self, offset):
        """Add lipid bonds and angles and return nonbonded exclusions."""
        exclusion_map = []  # for ah, yu etc.
        for i in range(0, self.nbeads - 1):
            for j in range(i, self.nbeads):
                if self.bond_check(i, j):
                    d = self.calc_bondlength(i, j)
                    if j - i == 1:
                        if self.molecule_type == "cooke_lipid":
                            kfene = 30 * 3 * self.eps_lj / d / d
                            bidx = self.wcafene.addBond(
                                i + offset,
                                j + offset,
                                [
                                    d * unit.nanometer,
                                    kfene
                                    * unit.kilojoules_per_mole
                                    / (unit.nanometer**2),
                                ],
                            )
                            self.bond_pairlist.append(
                                [i + offset + 1, j + offset + 1, bidx, d, kfene]
                            )  # 1-based
                        elif self.molecule_type == "lipid":
                            bidx = self.hb.addBond(
                                i + offset,
                                j + offset,
                                d * unit.nanometer,
                                1700 * unit.kilojoules_per_mole / (unit.nanometer**2),
                            )
                            self.bond_pairlist.append(
                                [i + offset + 1, j + offset + 1, bidx, d, 1700]
                            )  # 1-based
                        exclusion_map.append([i + offset, j + offset])
                    else:
                        if self.molecule_type == "cooke_lipid":
                            kbend = 30 * self.eps_lj / d / d
                            bidx = self.hb.addBond(
                                i + offset,
                                j + offset,
                                4 * d * unit.nanometer,
                                kbend * unit.kilojoules_per_mole / (unit.nanometer**2),
                            )
                            self.bond_pairlist.append(
                                [i + offset + 1, j + offset + 1, bidx, 4 * d, kbend]
                            )  # 1-based
                        else:
                            angle = 2 / 3 * np.pi if i == 0 else np.pi
                            k_angle = 7 / 2 if i == 0 else 7
                            self.ha.addAngle(
                                i + offset,
                                i + offset + 1,
                                j + offset,
                                angle * unit.radian,
                                k_angle * unit.kilojoules_per_mole / (unit.radian**2),
                            )
        return exclusion_map

    def get_forces(self):
        """Collect the forces required by the selected lipid model."""
        self.forces = [self.hb]
        if self.molecule_type == "lipid":
            self.forces.append(self.ha)
        elif self.molecule_type == "cooke_lipid":
            self.forces.append(self.wcafene)
        else:
            raise ValueError(f"Unknown lipid type {self.molecule_type}.")


class Crowder(Component):
    """Represent a generic linear coarse-grained crowder.

    Crowders use the shared component properties and consecutive bead bonds.
    """

    def calc_properties(
        self, pH: float = 7.0, verbose: bool = False,
    ):
        """Calculate crowder properties and initial coordinates."""

        super().calc_properties(pH=pH, verbose=verbose)
        self.calc_x_setup()  # can be overwritten in custom component

    @staticmethod
    def bond_check(i: int, j: int):
        """Return whether two crowder beads should be bonded."""

        condition = j == i + 1
        return condition


class Cyclic(Protein):
    """Represent a cyclic protein or peptide.

    The final bead is bonded back to the first bead to close the chain.
    """

    def bond_check(self, i: int, j: int):
        """Return whether two cyclic-chain beads should be bonded."""

        condition0 = j == i + 1
        condition1 = (j == self.nbeads - 1) and i == 0
        condition = condition0 or condition1
        return condition


class Seastar(Protein):
    """Represent a star-shaped branched protein or peptide.

    Branches extend from the first bead according to the configured end count.
    """

    def bond_check(self, i: int, j: int):
        """Return whether two branched-chain beads should be bonded."""

        if self.n_ends in [0, 1, 2]:
            return super().bond_check(i, j)
        else:
            if (self.nbeads - 1) % self.n_ends == 0:
                branch_length = int((self.nbeads - 1) / self.n_ends)
            else:
                branch_length = int((self.nbeads - 1) / self.n_ends) + 1

            condition0 = (j == i + 1) and ((j - 1) % branch_length != 0)
            condition1 = (i == 0) and ((j - 1) % branch_length == 0)

            condition = condition0 or condition1
            return condition


class PTMProtein(Protein):
    """Represent a protein with attached post-translational modifications.

    PTM sequences are appended as branches at configured protein residues.
    """

    def calc_comp_seq(self):
        """Calculate the combined protein and PTM sequence."""

        records = read_fasta(self.ffasta)
        self.seq = str(records[self.name].seq)  # one bead seq
        self.nbeads_protein = len(self.seq)
        self.ptm_seq = str(records[self.ptm_name].seq)

        for _ in self.ptm_locations:  # 1-based
            self.seq = self.seq + self.ptm_seq

        self.n_termini = [0]
        self.c_termini = [len(self.seq) - 1]

    def bond_check(self, i: int, j: int):
        """Return whether two protein or PTM beads should be bonded."""

        # residue-residue bond (protein)
        if (i < self.nbeads_protein - 1) and (j == i + 1):
            return True

        ptm_seqlocs = []
        # residue-PTM bond
        for idx, ptm_loc in enumerate(self.ptm_locations):
            ptm_seqloc = self.nbeads_protein + idx * len(
                self.ptm_seq
            )  # position of connecting PTM bead in sequence
            ptm_seqlocs.append(ptm_seqloc)
            if (i == ptm_loc - 1) and (j == ptm_seqloc):
                return True

        # PTM-PTM bond
        if i >= self.nbeads_protein:
            if (j == i + 1) and (j not in ptm_seqlocs):  # avoid bonding different PTMs
                return True
        return False

COMPONENT_REGISTRY: dict[str, type[Component]] = {
    "protein": Protein,
    "lipid": Lipid,
    "cooke_lipid": Lipid,
    "crowder": Crowder,
    "rna": RNA,
    "cyclic": Cyclic,
    "seastar": Seastar,
    "ptm_protein": PTMProtein,
}
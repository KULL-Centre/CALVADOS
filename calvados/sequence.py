"""Utilities for reading, manipulating, and characterizing biomolecular sequences."""

import os
import random
import warnings
from collections.abc import Iterable, Mapping, Sequence
from os import PathLike
from typing import Literal, TypeAlias

import numba as nb
import numpy as np
from Bio import SeqIO, SeqUtils
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from joblib import load
from MDAnalysis import Universe
from MDAnalysis.lib.util import convert_aa_code
from numpy.typing import NDArray
from openmm import app
from pandas import DataFrame
from scipy.integrate import quad

FloatArray: TypeAlias = NDArray[np.float64]
PairMap: TypeAlias = Mapping[tuple[str, str], float]
InputPath: TypeAlias = PathLike | str

### SEQUENCE INPUT / OUTPUT
def read_fasta(ffasta: InputPath) -> dict[str, SeqRecord]:
    """Read a FASTA file into a dictionary keyed by record ID."""
    return SeqIO.to_dict(SeqIO.parse(ffasta, "fasta"))


def seq_from_pdb(
    pdb: str,
    fmt: Literal["string", "list"] = "string",
) -> tuple[str | list[str], list[int], list[int]]:
    """Extract a sequence and terminal indices from a PDB or PDBx/mmCIF file."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if pdb.lower().endswith(".cif"):
            pdbx = app.pdbxfile.PDBxFile(pdb)
            u = Universe(pdbx)
        else:
            u = Universe(pdb)

    # we do not assume residues in the PDB file are numbered from 1
    n_termini = [0]
    c_termini = [len(u.atoms.segments[0].residues) - 1]
    for segment in u.atoms.segments[1:]:
        n_termini.append(c_termini[-1] + 1)
        c_termini.append(c_termini[-1] + len(segment.residues))

    ag = u.atoms
    if fmt not in ["string", "list"]:
        raise ValueError("fmt must be string or list.")
    sequence = []
    res3 = ag.residues.resnames
    for res in res3:
        if len(res) == 3:
            res1 = convert_aa_code(res)
        else:
            res1 = res
        if res1 == "":
            res1 = "X"
        sequence.extend(res1)
    fastapdb = "".join(sequence) if fmt == "string" else sequence
    return fastapdb, n_termini, c_termini


def write_fasta(new_records: Iterable[SeqRecord], fout: InputPath) -> None:
    """Append records with new IDs to a FASTA file, or create the file."""
    if not os.path.isfile(fout):
        SeqIO.write(new_records, fout, "fasta")
        return

    records = list(SeqIO.parse(fout, "fasta"))
    ids = {record.id for record in records}
    records.extend(record for record in new_records if record.id not in ids)
    SeqIO.write(records, fout, "fasta")


def record_from_seq(seq: str, name: str) -> SeqRecord:
    """Create a minimally annotated sequence record."""
    return SeqRecord(Seq(seq), id=name, name="", description="")


### SEQUENCE ANALYSIS


def get_qs(
    seq: str,
    flexhis: bool = False,
    pH: float = 7.0,
    residues: DataFrame | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Calculate signed and absolute residue charges for a sequence."""
    if residues is None:
        return get_qs_fast(seq, flexhis=flexhis, pH=pH)

    charges: list[float] = []
    for s in seq:
        if flexhis and s == "H":
            q = 1.0 / (1 + 10 ** (pH - 6))
        else:
            q = residues.loc[s].q
        charges.append(q)
    qs: FloatArray = np.asarray(charges, dtype=np.float64)
    return qs, np.abs(qs)


@nb.jit(nopython=True)
def get_qs_fast(
    seq: str,
    flexhis: bool = False,
    pH: float = 7.0,
    phos_negative: bool = True,
) -> tuple[FloatArray, FloatArray]:
    """Calculate residue charges without consulting a residue parameter table."""
    qs: FloatArray = np.zeros(len(seq), dtype=np.float64)
    list_negative = ["E", "D", "p"] if phos_negative else ["E", "D"]

    for idx in range(len(seq)):
        if seq[idx] in ["R", "K"]:
            qs[idx] = 1.0
        elif seq[idx] in list_negative:
            qs[idx] = -1.0
        elif seq[idx] == "H" and flexhis:
            qs[idx] = 1.0 / (1 + 10 ** (pH - 6))
    return qs, np.abs(qs)


def patch_terminal_qs(
    qs: FloatArray,
    n_termini: Sequence[int] | NDArray[np.int64],
    c_termini: Sequence[int] | NDArray[np.int64],
    loc: Literal["N", "C", "both"] = "both",
) -> FloatArray:
    """Add charges to selected N- and C-termini."""
    qsnew = qs.copy()

    if loc in ["N", "both"]:
        qsnew[n_termini] += 1.0
    if loc in ["C", "both"]:
        qsnew[c_termini] -= 1.0
    return qsnew


def patch_terminal_mws(
    mws: FloatArray,
    n_termini: Sequence[int] | NDArray[np.int64],
    c_termini: Sequence[int] | NDArray[np.int64],
    loc: Literal["N", "C", "both"] = "both",
) -> FloatArray:
    """Add terminal hydrogen and oxygen masses to residue masses."""
    mwsnew = mws.copy()
    if loc in ["N", "both"]:
        mwsnew[n_termini] += 2
    if loc in ["C", "both"]:
        mwsnew[c_termini] += 16
    return mwsnew


def seq_dipole(seq: str) -> tuple[float, float]:
    """Calculate the center of charge and one-dimensional sequence dipole."""
    qs, qs_abs = get_qs(seq)
    com = seq_com(qs_abs)
    dip = 0.0

    for idx, q in enumerate(qs):
        dip += (com - idx) * q  # positive if positive towards Nterm
    return com, dip


def seq_com(qs_abs: Sequence[float] | FloatArray) -> float:
    """Calculate the center of the absolute charges along a sequence."""
    com = 0.0
    ct = 0.0
    for idx, q_abs in enumerate(qs_abs):
        ct += q_abs
        com += q_abs * idx
    if ct > 0:
        return com / ct
    return len(qs_abs) // 2


@nb.jit(nopython=True)
def calc_SCD(qs: FloatArray) -> float:
    """Calculate sequence charge decoration as defined by Sawle and Ghosh."""
    N = len(qs)
    scd = 0.0
    for idx in range(1, N):
        for jdx in range(idx):
            scd += qs[idx] * qs[jdx] * (idx - jdx) ** 0.5
    return scd / N


def calc_SHD(seq: str, lambda_map: PairMap, beta: float = -1.0) -> float:
    """Calculate sequence hydropathy decoration as defined by Zheng et al."""
    N = len(seq)
    shd = 0.0

    for idx in range(N - 1):
        seqi = seq[idx]
        for jdx in range(idx + 1, N):
            seqj = seq[jdx]
            shd += lambda_map[(seqi, seqj)] * (jdx - idx) ** beta
    return shd / N


def mean_lambda(seq: str, residues: DataFrame) -> float:
    """Calculate the mean hydropathy parameter of a sequence."""
    lambdas_sum = 0.0
    for x in seq:
        lambdas_sum += residues.lambdas[x]
    return lambdas_sum / len(seq)


def calc_aromatics(seq: str | Seq) -> tuple[float, float, float]:
    """Calculate the tyrosine, phenylalanine, and tryptophan fractions."""
    seq = str(seq)
    N = len(seq)
    return seq.count("Y") / N, seq.count("F") / N, seq.count("W") / N


def calc_mw(fasta: Iterable[str], residues: DataFrame | None = None) -> float:
    """Calculate the molecular weight of a sequence in daltons."""
    seq = "".join(fasta)
    if residues is None:
        return SeqUtils.molecular_weight(seq, seq_type="protein")

    mw = 0.0
    for s in seq:
        mw += residues.loc[s, "MW"]
    return mw


### SEQUENCE MANIPULATION
def shuffle_str(seq: str | Sequence[str]) -> str:
    """Return a randomly shuffled copy of a sequence."""
    l = list(seq)
    random.shuffle(l)
    return "".join(l)


def construct_maxdipseq(seq: str) -> str:
    """Construct a sequence permutation with maximal charge separation."""
    seqpos, seqneg, seqneu = split_seq(seq)
    return seqpos + seqneu + seqneg


def split_seq(seq: str) -> tuple[str, str, str]:
    """Split a sequence into shuffled positive, negative, and neutral groups."""
    seqpos = []
    seqneg = []
    seqneu = []
    for s in seq:
        if s in ["K", "R"]:
            seqpos.append(s)
        elif s in ["D", "E"]:
            seqneg.append(s)
        else:
            seqneu.append(s)
    return shuffle_str(seqpos), shuffle_str(seqneg), shuffle_str(seqneu)


@nb.jit(nopython=True)
def lj_potential(r: float, sig: float, eps: float) -> float:
    """Evaluate the Lennard-Jones potential."""
    return 4.0 * eps * ((sig / r) ** 12 - (sig / r) ** 6)


@nb.jit(nopython=True)
def ah_potential(r: float, sig: float, eps: float, l: float, rc: float) -> float:
    """Evaluate the shifted Ashbaugh-Hatch potential."""
    if r <= 2 ** (1.0 / 6.0) * sig:
        return (
            lj_potential(r, sig, eps) - l * lj_potential(rc, sig, eps) + eps * (1 - l)
        )
    if r <= rc:
        return l * (lj_potential(r, sig, eps) - lj_potential(rc, sig, eps))
    return 0.0


def ah_scaled(r: float, sig: float, eps: float, l: float, rc: float) -> float:
    """Scale the Ashbaugh-Hatch potential by the spherical volume element."""
    return ah_potential(r, sig, eps, l, rc) * 4 * np.pi * r**2


def make_ah_intgrl_map(
    residues: DataFrame, rc: float = 2.0, eps: float = 0.2 * 4.184
) -> dict[tuple[str, str], float]:
    """Integrate the Ashbaugh-Hatch potential for each residue pair."""
    ah_intgrl_map = {}
    for key0, val0 in residues.iterrows():
        sig0, l0 = val0["sigmas"], val0["lambdas"]
        for key1, val1 in residues.iterrows():
            sig1, l1 = val1["sigmas"], val1["lambdas"]
            sig, l = 0.5 * (sig0 + sig1), 0.5 * (l0 + l1)
            integral = quad(
                lambda r, sig=sig, l=l: ah_scaled(r, sig, eps, l, rc),
                2 ** (1.0 / 6.0) * sig,
                rc,
            )[0]
            ah_intgrl_map[(key0, key1)] = integral
            ah_intgrl_map[(key1, key0)] = integral
    return ah_intgrl_map


def make_lambda_map(residues: DataFrame) -> dict[tuple[str, str], float]:
    """Build a map of summed hydropathy parameters for residue pairs."""
    lambda_map = {}
    for key0, val0 in residues.iterrows():
        l0 = val0["lambdas"]
        for key1, val1 in residues.iterrows():
            l1 = val1["lambdas"]
            l = l0 + l1
            lambda_map[(key0, key1)] = l
            lambda_map[(key1, key0)] = l
    return lambda_map


def calc_ah_ij(seq: str, ah_intgrl_map: PairMap) -> float:
    """Calculate the mean integrated Ashbaugh-Hatch interaction of a sequence."""
    U = 0.0
    N = len(seq)
    for idx in range(N):
        seqi = seq[idx]
        for jdx in range(idx, N):
            U += ah_intgrl_map[(seqi, seq[jdx])]
    U /= N * (N - 1) / 2.0 + N
    return U


############ FAST KAPPA ################


@nb.jit(nopython=True)
def check_dmax(seq: str, dmax: float, seqmax: str) -> tuple[str, float]:
    """Keep a sequence when its charge asymmetry exceeds the current maximum."""
    qs, _ = get_qs_fast(seq)
    d = calc_delta(qs)
    if d > dmax:
        return seq, d
    return seqmax, dmax


@nb.jit(nopython=True)
def calc_case0(seqpos: str, seqneg: str, seqneu: str) -> str:
    """Construct the maximum-delta sequence when only one charge sign is present."""
    seqmax = ""
    dmax = 0.0
    N = len(seqpos) + len(seqneg) + len(seqneu)
    if len(seqpos) == 0:
        seqcharge = seqneg[:]
    elif len(seqneg) == 0:
        seqcharge = seqpos[:]
    if len(seqneu) > len(seqcharge):
        for pos in range(N - len(seqcharge) + 1):
            seqout = seqneu[:pos] + seqcharge + seqneu[pos:]
            seqmax, dmax = check_dmax(seqout, dmax, seqmax)
    else:
        for pos in range(N - len(seqneu) + 1):
            seqout = seqcharge[:pos] + seqneu + seqcharge[pos:]
            seqmax, dmax = check_dmax(seqout, dmax, seqmax)
    return seqmax


@nb.jit(nopython=True)
def calc_case1(seqpos: str, seqneg: str, seqneu: str) -> str:
    """Construct the maximum-delta sequence when no neutral residues are present."""
    seqmax = ""
    dmax = 0.0
    N = len(seqpos) + len(seqneg) + len(seqneu)
    if len(seqpos) > len(seqneg):
        for pos in range(N - len(seqneg) + 1):
            seqout = seqpos[:pos] + seqneg + seqpos[pos:]
            seqmax, dmax = check_dmax(seqout, dmax, seqmax)
    else:
        for neg in range(N - len(seqpos) + 1):
            seqout = seqneg[:neg] + seqpos + seqneg[neg:]
            seqmax, dmax = check_dmax(seqout, dmax, seqmax)
    return seqmax


@nb.jit(nopython=True)
def calc_case2(seqpos: str, seqneg: str, seqneu: str) -> str:
    """Approximate the maximum-delta sequence for at least 18 neutral residues."""
    seqmax = ""
    dmax = 0.0
    for startNeuts in range(7):
        for endNeuts in range(7):
            startBlock = seqneu[:startNeuts]
            endBlock = seqneu[startNeuts : startNeuts + endNeuts]
            midBlock = seqneu[startNeuts + endNeuts :]

            seqout = startBlock + seqpos + midBlock + seqneg + endBlock
            seqmax, dmax = check_dmax(seqout, dmax, seqmax)
    return seqmax


@nb.jit(nopython=True)
def calc_case3(seqpos: str, seqneg: str, seqneu: str) -> str:
    """Construct the maximum-delta sequence for fewer than 18 neutral residues."""
    seqmax = ""
    dmax = 0.0
    for midNeuts in range(len(seqneu) + 1):
        midBlock = seqneu[:midNeuts]
        for startNeuts in range(len(seqneu) - midNeuts + 1):
            startBlock = seqneu[midNeuts : midNeuts + startNeuts]
            seqout = (
                startBlock
                + seqpos
                + midBlock
                + seqneg
                + seqneu[midNeuts + startNeuts :]
            )
            seqmax, dmax = check_dmax(seqout, dmax, seqmax)
    return seqmax


def construct_deltamax(seq: str) -> str:
    """Construct a sequence permutation with maximal charge asymmetry."""
    seqpos, seqneg, seqneu = split_seq(seq)

    if (len(seqpos) == 0) or (len(seqneg) == 0):
        return calc_case0(seqpos, seqneg, seqneu)
    if len(seqneu) == 0:
        return calc_case1(seqpos, seqneg, seqneu)
    if len(seqneu) >= 18:
        return calc_case2(seqpos, seqneg, seqneu)
    return calc_case3(seqpos, seqneg, seqneu)


def calc_kappa_manual(seq: str) -> float:
    """Calculate the sequence charge-patterning parameter kappa."""
    qs, qs_abs = get_qs_fast(seq, phos_negative=False)
    if np.sum(qs_abs) == 0:
        return -1

    seqpos, seqneg, seqneu = split_seq(seq)
    if len(seqneu) == 0 and (len(seqneg) == 0 or len(seqpos) == 0):
        return -1

    delta = calc_delta(qs)

    seq_max = construct_deltamax(seq)
    qs_max, _ = get_qs_fast(seq_max, phos_negative=False)
    delta_max = calc_delta(qs_max)

    return delta / delta_max


@nb.jit(nopython=True)
def calc_delta(qs: FloatArray) -> float:
    """Calculate mean charge asymmetry over five- and six-residue windows."""
    return (calc_delta_form(qs, window=5) + calc_delta_form(qs, window=6)) / 2.0


@nb.jit(nopython=True)
def calc_delta_form(qs: FloatArray, window: int = 5) -> float:
    """Calculate charge asymmetry for a given sliding-window size."""
    sig_m = calc_sigma(qs)

    nw = len(qs) - window + 1
    sigs = np.zeros(nw)

    for idx in range(nw):
        sigs[idx] = calc_sigma(qs[idx : idx + window])
    return np.sum((sigs - sig_m) ** 2) / nw


@nb.jit(nopython=True)
def frac_charges(qs: FloatArray) -> tuple[float, float]:
    """Calculate positive and negative fractions, excluding fractional charges."""
    N = len(qs)
    fpos = 0.0
    fneg = 0.0
    for idx in range(N):
        if qs[idx] >= 1:
            fpos += 1.0
        elif qs[idx] <= -1:
            fneg += 1.0
    return fpos / N, fneg / N


@nb.jit(nopython=True)
def calc_sigma(qs: FloatArray) -> float:
    """Calculate charge asymmetry from positive and negative charge fractions."""
    fpos, fneg = frac_charges(qs)
    ncpr = fpos - fneg
    fcr = fpos + fneg
    if fcr == 0:
        return 0.0
    return ncpr**2 / fcr


class SeqFeatures:
    """Calculate physicochemical features for a biomolecular sequence.

    Charge-based attributes are always calculated. Hydropathy, molecular weight,
    and integrated Ashbaugh-Hatch attributes are added when a residue parameter
    table is supplied. If ``nu_file`` is provided, the class also calculates
    kappa and predicts the scaling exponent using the stored model.

    Parameters
    ----------
    seq
        One-letter residue sequence.
    residues
        Optional residue parameter table indexed by one-letter residue code.
    charge_termini
        Whether to add unit charges to the N- and C-termini.
    nu_file
        Optional path to a serialized scaling-exponent prediction model.
    ah_intgrl_map, lambda_map
        Optional precomputed residue-pair maps.
    flexhis, pH
        Whether and at which pH to calculate fractional histidine charges.
    """

    def __init__(
        self,
        seq: str,
        residues: DataFrame | None = None,
        charge_termini: bool = False,
        nu_file: InputPath | None = None,
        ah_intgrl_map: PairMap | None = None,
        lambda_map: PairMap | None = None,
        flexhis: bool = False,
        pH: float = 7.0,
    ) -> None:
        """Initialize and calculate the requested sequence features."""
        self.seq = seq
        self.N = len(seq)
        self.qs, self.qs_abs = get_qs(seq, flexhis=flexhis, pH=pH, residues=residues)
        if charge_termini:
            self.qs[0] += 1.0
            self.qs[-1] -= 1.0
        self.charge = np.sum(self.qs)
        self.fpos, self.fneg = frac_charges(self.qs)
        self.ncpr = self.charge / self.N
        self.fcr = self.fpos + self.fneg
        self.scd = calc_SCD(self.qs)
        self.rY, self.rF, self.rW = calc_aromatics(seq)
        self.faro = self.rY + self.rF + self.rW

        if residues is not None:
            self.lambdas_mean = mean_lambda(seq, residues)
            self.mean_lambda = self.lambdas_mean
            if lambda_map is None:
                lambda_map = make_lambda_map(residues)
            self.shd = calc_SHD(seq, lambda_map, beta=-1.0)
            self.mw = calc_mw(seq, residues=residues)
            if ah_intgrl_map is None:
                ah_intgrl_map = make_ah_intgrl_map(residues)
            self.ah_ij = calc_ah_ij(seq, ah_intgrl_map)

        if nu_file is not None:
            self.kappa = calc_kappa_manual(seq)
            if self.kappa == -1:  # no charges
                self.kappa = 0.0
            if flexhis:
                self.qs_noflex, _ = get_qs(seq, residues=residues, flexhis=False)
                if charge_termini:
                    self.qs_noflex[0] += 1.0
                    self.qs_noflex[-1] -= 1.0
                self.scd_noflex = calc_SCD(self.qs_noflex)
            else:
                self.scd_noflex = self.scd
            feats_for_nu = [
                self.scd_noflex,
                self.shd,
                self.kappa,
                self.fcr,
                self.mean_lambda,
            ]
            model_nu = load(nu_file)
            X_nu: FloatArray = np.reshape(
                np.array(feats_for_nu, dtype=np.float64), (1, -1)
            )
            self.nu_svr = model_nu.predict(X_nu)[0]

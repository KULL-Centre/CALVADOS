from json import load
from os import PathLike
from typing import Literal, Sequence
from warnings import catch_warnings, simplefilter

import numpy as np
import pandas as pd
from MDAnalysis import Universe
from MDAnalysis.analysis import distances
from numpy.typing import NDArray
from openmm import app, unit
from scipy import constants
from yaml import safe_load

from .sequence import calc_mw


################ SYSTEM BUILDING FUNCTIONS ################


def build_box(
    Lx: float, Ly: float, Lz: float
) -> tuple[unit.Quantity, unit.Quantity, unit.Quantity]:
    """Build orthogonal OpenMM box vectors with lengths in nm."""
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = Lx * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = Ly * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = Lz * unit.nanometers
    return a, b, c


def calc_box(N: int) -> list[float]:
    """Select default box lengths in nm based on chain length."""
    if N > 350:
        return [25.0, 25.0, 300.0]
    return [20.0, 20.0, 200.0]


def calc_nprot_slab(N: int, box: NDArray[np.float64], pbeads: float = 90) -> int:
    """Calculate proteins from chain length and bead density in nm^-2."""
    beads = pbeads * box[0] * box[1]
    return int(beads / N)


def check_walls(x: NDArray[np.float64], box: NDArray[np.float64]) -> bool:
    """Check whether any coordinate lies outside the box."""
    return bool(np.any(x < 0) or np.any(x > box))


def check_clash(
    x: NDArray[np.float64],
    pos: NDArray[np.float64],
    box: NDArray[np.float64],
    cutoff: float = 0.7,
) -> bool:
    """Check for a periodic interparticle distance below the cutoff in nm."""
    if pos.size == 0:
        return False
    boxfull = np.append(box, [90, 90, 90])
    d = distances.distance_array(x, pos, boxfull)
    return bool(np.amin(d) < cutoff)


def draw_vec(length: float, ndim: int = 3) -> NDArray[np.float64]:
    """Draw an unbiased random vector of the requested length."""
    while True:
        vec = np.random.random(size=ndim) - 0.5
        norm = np.linalg.norm(vec)
        if 0 < norm < 0.5:
            break
    return vec / norm * length


def draw_starting_vec(box: NDArray[np.float64]) -> NDArray[np.float64]:
    """Draw a random position within the simulation box."""
    return np.random.random(size=3) * box


def build_linear(
    z_bondlengths: NDArray[np.float64],
    n_per_res: int = 1,
    ys: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Build a linear chain centred at the origin."""
    N = len(z_bondlengths)
    zs = np.zeros(N)
    zs[1:] = [(z_bondlengths[idx] + z_bondlengths[idx + 1]) / 2 for idx in range(N - 1)]
    zs = np.cumsum(zs)
    coords = np.zeros((N * n_per_res, 3))
    if ys is None:
        ys = np.zeros(N)
    if len(ys) != len(zs):
        raise ValueError("Length of y offsets must be equal to bead counts")
    for idx, (z, y) in enumerate(zip(zs, ys)):
        for jdx in range(n_per_res):
            coords[idx * n_per_res + jdx] = [0, jdx * y, z]
    coords[:, 2] -= coords[:, 2].mean()
    return coords


def p2c(r: float, phi: float) -> tuple[float, float]:
    """Convert polar coordinates to Cartesian coordinates."""
    return (r * np.cos(phi), r * np.sin(phi))


def build_spiral(
    bondlengths: NDArray[np.float64],
    delta: float | NDArray[np.float64] = 0,
    arc: float = 0.38,
    separation: float = 0.7,
    n_per_res: int = 1,
) -> NDArray[np.float64]:
    """Build points on an Archimedean spiral."""
    r = arc
    b = separation / (2 * np.pi)
    phi = float(r) / b
    coords = []
    for z in bondlengths:
        for j in range(
            n_per_res
        ):  # number of beads per residue (placed along z with bondlength)
            coords.append(list(p2c(r, phi)) + [j * z])  # j*z = 0 for n_per_res=1
        phi += float(arc) / r
        r = b * phi
    return np.array(coords) + delta


def build_compact(
    nbeads: int, d: float = 0.38, verbose: bool = False
) -> NDArray[np.float64]:
    """Build a compact serpentine cubic grid with spacing in nm."""
    if not nbeads > 0:
        raise ValueError("nbeads must be greater than 0")
    N = int(np.ceil(np.cbrt(nbeads)) - 1)
    if verbose:
        print(f"Building {N + 1} * {N + 1} * {N + 1} grid.")
    xs = []
    i, j, k = 0, 0, 0
    di, dj, dk = 1, 1, 1  # direction
    ctj, ctk = 0, 0

    for _ in range(nbeads):
        xs.append([i, j, k])
        if ctk == N:
            if ctj == N:
                i += di
                ctj = 0
                dj *= -1
            else:
                j += dj
                ctj += 1
            ctk = 0
            dk *= -1
        else:
            k += dk
            ctk += 1
    coords = (np.array(xs) - 0.5 * N) * d
    return coords


def random_placement(
    box: NDArray[np.float64],
    xs_others: NDArray[np.float64],
    xinit: NDArray[np.float64],
    ntries: int = 10000,
) -> NDArray[np.float64]:
    """Place a molecule randomly without wall or particle clashes."""
    for _ in range(ntries):
        x0 = draw_starting_vec(box)  # random point in box
        xs = x0 + xinit
        walls = check_walls(xs, box)  # check if outside box
        if walls:
            continue
        clash = check_clash(xs, xs_others, box)  # check if clashes with existing pos
        if not clash:
            return xs
    raise ValueError(f"Tried {ntries}x to add molecule. Giving up.")


def build_xybilayer(
    x0: NDArray[np.float64],
    box: NDArray[np.float64],
    xs_others: NDArray[np.float64],
    xinit: NDArray[np.float64],
    upward: bool = True,
) -> tuple[NDArray[np.float64], bool]:
    """Place a molecule in an xy bilayer."""
    inserted = True
    xs = x0 + xinit
    xs[:, 2] -= xs[1, 2]
    xs[:, 2] += box[2] / 2 - 1.5
    if upward:
        xs = xs[::-1, :]
        xs[:, 2] += 3
    walls = check_walls(xs, box)  # check if outside box
    if walls:
        inserted = False
    clash = check_clash(
        xs, xs_others, box, cutoff=0.5
    )  # check if clashes with existing pos
    if clash:
        inserted = False
    return xs, inserted


def build_xygrid(
    N: int, box: NDArray[np.float64], z: float = 0.0
) -> NDArray[np.float64]:
    """Build an xy grid at a fixed z coordinate."""
    if np.sqrt(N) % 1 > 0:
        b = 2
    else:
        b = 1
    nx = int(np.sqrt(N)) + b  # nx spots in x dim
    ny = int(np.sqrt(N)) + b  # ny spots in x dim

    dx = box[0] / nx
    dy = box[1] / ny

    xy = []
    x, y = 0.0, 0.0
    ct = 0
    for _ in range(N):
        ct += 1
        xy.append([x, y, z])
        if ct == ny:
            y = 0
            x += dx
            ct = 0
        else:
            y += dy
    return np.array(xy)


def build_xyzgrid(N: int | float, box: NDArray[np.float64]) -> NDArray[np.float64]:
    """Build a staggered three-dimensional grid."""
    r = box / np.sum(box)
    a = np.cbrt(N / np.prod(r))
    n = a * r
    nxyz = np.floor(n)
    while np.prod(nxyz) < N:
        ndeviation = n / nxyz
        devmax = np.argmax(ndeviation)
        nxyz[devmax] += 1
    while np.prod(nxyz) > N:
        nmax = np.argmax(nxyz)
        nxyz[nmax] -= 1
        if np.prod(nxyz) < N:
            nxyz[nmax] += 1
            break

    xyz = []
    x, y, z = 0.0, 0.0, 0.0

    ctx, cty = 0, 0

    dx = box[0] / nxyz[0]
    dy = box[1] / nxyz[1]
    dz = box[2] / nxyz[2]

    zplane = 1
    xyplane = 1

    for _ in np.arange(N):
        if zplane > 0:
            xshift = 0
            yshift = 0
        else:
            xshift = dx / 2
            yshift = dy / 2

        if xyplane < 0:
            zshift = dz / 2
        else:
            zshift = 0

        xyz.append([x + xshift, y + yshift, z + zshift])

        ctx += 1
        x += dx

        if ctx % 2 == cty % 2:
            xyplane = 1
        else:
            xyplane = -1

        if ctx == nxyz[0]:
            ctx = 0
            x = 0

            cty += 1
            y += dy

            if cty == nxyz[1]:
                ctx = 0
                cty = 0
                x = 0.0
                y = 0.0

                z += dz

                zplane = -zplane

    return np.asarray(xyz)


# FOLDED
def geometry_from_pdb(
    pdb: str | PathLike[str], use_com: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Return protein coordinates and box lengths converted from Å to nm."""
    pdb = str(pdb)
    with catch_warnings():
        simplefilter("ignore")
        if pdb.lower().endswith(".cif"):
            pdbx = app.pdbxfile.PDBxFile(pdb)
            u = Universe(pdbx)
        else:
            u = Universe(pdb)
    ag = u.atoms
    ag.translate(-ag.center_of_mass())
    if use_com:
        coms = [res.atoms.center_of_mass() for res in u.residues]
        pos = np.array(coms) / 10.0
    else:
        cas = u.select_atoms("name CA")
        pos = cas.positions / 10.0
    if u.dimensions is None:
        box = None
    else:
        box = np.append(u.dimensions[:3] / 10.0, u.dimensions[3:])
    return pos, box


def geometry_from_pdb_rna(
    pdb: str | PathLike[str], use_com: bool = False
) -> tuple[NDArray[np.float64], NDArray[np.float64] | None]:
    """Return RNA coordinates and box lengths converted from Å to nm."""
    backbone_atoms_name = [
        "1H2'",
        "1H5'",
        "2H5'",
        "2HO'",
        "C1'",
        "C2'",
        "C3'",
        "C4'",
        "C5'",
        "H1'",
        "H3'",
        "H4'",
        "O2'",
        "O3'",
        "O5'",
        "O4'",
        "OP1",
        "OP2",
        "P",
    ]

    with catch_warnings():
        simplefilter("ignore")
        u = Universe(str(pdb))
    ag = u.atoms
    ag.translate(-ag.center_of_mass())
    positions = []
    if use_com:
        for res in u.residues:
            backbone = res.atoms.select_atoms("name " + " ".join(backbone_atoms_name))
            non_backbone = res.atoms.difference(backbone)
            positions.append(backbone.center_of_mass())
            positions.append(non_backbone.center_of_mass())

    else:
        for res in u.residues:
            ps = res.atoms.select_atoms("name P")
            positions.append(ps.positions[0])
            if res.resname in ["U", "C", "T"]:
                ns = res.atoms.select_atoms("name N1")
                positions.append(ns.positions[0])
            elif res.resname in ["A", "G"]:
                ns = res.atoms.select_atoms("name N9")
                positions.append(ns.positions[0])
            else:
                raise ValueError(f"Invalid RNA resname, {res.resname}")
    pos = np.asarray(positions, dtype=float) / 10.0

    if u.dimensions is None:
        box = None
    else:
        box = np.append(u.dimensions[:3] / 10.0, u.dimensions[3:])

    return pos, box


def bfac_from_pdb(
    pdb: str | PathLike[str], confidence: float = 70.0
) -> NDArray[np.float64]:
    """Return residue-averaged pLDDT values above a confidence threshold."""
    with catch_warnings():
        simplefilter("ignore")
        u = Universe(str(pdb))
    bfac = np.zeros((len(u.residues)))
    for idx, res in enumerate(u.residues):
        bfac[idx] = np.mean(res.atoms.tempfactors)  # average b-factor for residue
    bfac = np.where(bfac > confidence, bfac, 0.0) / 100.0  # high confidence filter
    return bfac


def load_pae_inv(
    input_pae: str | PathLike[str],
    cutoff: float = 0.1,
    colabfold: Literal[0, 1, 2] = 0,
    symmetrize: bool = True,
) -> NDArray:
    """Load an AlphaFold PAE matrix and return its thresholded inverse."""
    pae = load_pae(input_pae, colabfold=colabfold, symmetrize=symmetrize)
    pae = np.where(pae < 1.0, 1, pae)  # avoid division by zero (for i = j), min to 1
    pae_inv = 1 / pae  # inverse pae
    pae_inv = np.where(pae_inv > cutoff, pae_inv, 0)
    return pae_inv


def load_pae(
    input_pae: str | PathLike[str],
    colabfold: Literal[0, 1, 2] = 0,
    symmetrize: bool = True,
) -> NDArray:
    """Load an AlphaFold PAE matrix from JSON."""
    if colabfold not in [0, 1, 2]:
        raise ValueError("colabfold must be 0, 1, or 2.")
    with open(input_pae) as f:
        data = load(f)
        if colabfold == 0:
            pae = np.array(data[0]["predicted_aligned_error"])
        elif colabfold == 1:
            pae = np.array(data["predicted_aligned_error"])
        elif colabfold == 2:
            pae = np.array(data["pae"])
    if symmetrize:
        pae = 0.5 * (pae + pae.T)
    return pae


def get_ssdomains(
    name: str, fdomains: str | PathLike[str], dpam: bool = False
) -> list[list[int]]:
    """Load structured domains and convert residue indices to zero-based."""
    if dpam:
        domains = []
        df_dpam = pd.read_csv(fdomains, delimiter="\t").set_index("uniprot")
        for key, val in df_dpam.iterrows():
            if key == name:
                rng = val["range"].split("-")
                domains.append([int(rng[0]), int(rng[1])])
    else:
        with open(fdomains) as f:
            stream = f.read()
            domainbib = safe_load(stream)
        domains = domainbib[name]

    print(f"Using domains {domains}")

    ssdomains = []
    for domain in domains:
        xs = []  # restraint residues of domain
        if isinstance(domain[0], list):
            for subdom in domain:
                for x in range(subdom[0] - 1, subdom[1]):
                    xs.append(x)
        else:
            for x in range(domain[0] - 1, domain[1]):
                xs.append(x)
        ssdomains.append(xs)  # converted from 1-based inclusive ranges
    return ssdomains


def check_ssdomain(
    ssdomains: Sequence[Sequence[int]], i: int, j: int, req_both: bool = True
) -> bool:
    """Check whether one or both zero-based residues share a domain."""
    for ssdom in ssdomains:
        if req_both:
            if (i in ssdom) and (j in ssdom):
                return True
        else:
            if (i in ssdom) or (j in ssdom):
                return True
    return False


def conc_to_n(cinp: float, p: float, V: float, mw: float) -> float:
    """Convert concentration in g/L and volume in nm³ to molecule count."""
    return p * 1e-24 * 1 / mw * cinp * V * constants.N_A


def n_to_conc(n: float, V: float, mw: float) -> float:
    """Convert molecule count and volume in nm³ to concentration in g/L."""
    return n * mw * 1 / constants.N_A * 1e24 * 1 / V


def calc_pair_n_in_box(
    cinp: float, pB: float, box: NDArray[np.float64], seqA: str, seqB: str
) -> tuple[int, int, float, float]:
    """Calculate molecule counts for two species in a box measured in nm."""
    pA = 1.0 - pB
    V = box[0] * box[1] * box[2]  # nm^3
    mwA = calc_mw(seqA)  # Da = g/mol
    mwB = calc_mw(seqB)  # Da = g/mol

    nA = round(conc_to_n(cinp, pA, V, mwA))
    nB = round(conc_to_n(cinp, pB, V, mwB))

    mA_rounded = nA * mwA
    mB_rounded = nB * mwB

    pB_rounded = mB_rounded / (mA_rounded + mB_rounded)
    c_rounded = n_to_conc(nA, V, mwA) + n_to_conc(nB, V, mwB)
    return nA, nB, pB_rounded, c_rounded


def calc_mixture_n_in_box(
    cinp: float,
    ps: Sequence[float],
    box: NDArray[np.float64],
    seqs: Sequence[str],
) -> tuple[list[int], NDArray[np.float64], np.float64]:
    """Calculate mixture molecule counts in a box measured in nm."""
    V = box[0] * box[1] * box[2]  # nm^3

    ns = []
    mws = []

    for p, seq in zip(ps, seqs):
        mw = calc_mw(seq)  # Da = g/mol
        mws.append(mw)
        n = round(conc_to_n(cinp, p, V, mw))  # number of proteins per type
        ns.append(n)

    cs_rounded = np.array(
        [n_to_conc(n, V, mw) for n, mw in zip(ns, mws)]
    )  # rounded g/L per type
    ctotal: np.float64 = np.sum(cs_rounded)  # total g/L mass conc
    ps_rounded = cs_rounded / ctotal  # mass fractions

    return ns, ps_rounded, ctotal

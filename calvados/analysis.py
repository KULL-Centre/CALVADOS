import os
from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import matplotlib.pyplot as plt
import MDAnalysis as mda
import mdtraj as md
import numba as nb
import numpy as np
import pandas as pd
from Bio import SeqUtils
from MDAnalysis import Merge, transformations
from MDAnalysis.analysis import distances, rms
from MDAnalysis.analysis.align import AlignTraj
from numpy.typing import NDArray
from scipy.optimize import curve_fit, least_squares
from scipy.stats import sem
from tqdm import tqdm

from .BLOCKING.main import BlockAnalysis
from .build import get_ssdomains
from .inputmodels import InputPath

FloatArray: TypeAlias = NDArray[np.float64]
IntArray: TypeAlias = NDArray[np.int_]
ChainRange: TypeAlias = int | tuple[int, int]
ChainIds: TypeAlias = ChainRange | IntArray
ConcentrationResults: TypeAlias = dict[str, float | FloatArray]


def center_traj(
    pdb: InputPath,
    traj: InputPath,
    start: int | None = None,
    stop: int | None = None,
    step: int = 1,
) -> None:
    """Center each trajectory frame in the periodic box.

    Writes ``<traj>_c.dcd`` beside the input trajectory and returns nothing.
    """
    pdb = os.fspath(pdb)
    traj = os.fspath(traj)
    u = mda.Universe(pdb,traj)
    with mda.Writer(f'{traj[:-4]}_c.dcd', len(u.atoms)) as W:
        for ts in u.trajectory[start:stop:step]:
            u.atoms.translate(-u.atoms.center_of_geometry() + 0.5 * u.dimensions[:3])
            W.write(u.atoms)

def subsample_traj(
    pdb: InputPath,
    traj: InputPath,
    start: int | None = None,
    stop: int | None = None,
    step: int = 1,
) -> None:
    """Write a selected range of frames to ``<traj>_sub.dcd``."""
    pdb = os.fspath(pdb)
    traj = os.fspath(traj)

    u = mda.Universe(pdb,traj)

    with mda.Writer(f'{traj[:-4]}_sub.dcd', len(u.atoms)) as W:
        for ts in u.trajectory[start:stop:step]:
            W.write(u.atoms)

@nb.jit(nopython=True)
def calc_energy(
    dmap: FloatArray,
    sig: FloatArray,
    lam: FloatArray,
    rc_lj: float,
    eps_lj: float,
    qmap: FloatArray,
    k_yu: float,
    rc_yu: float = 4.0,
    same_domain: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """Calculate pairwise Ashbaugh-Hatch and Yukawa energies.

    Returns two arrays shaped like ``dmap``: Ashbaugh-Hatch energies first and
    Yukawa energies second, in the units implied by the supplied prefactors.
    """
    u_ah = np.zeros(dmap.shape)
    u_yu = np.zeros(dmap.shape)
    for i in range(dmap.shape[0]):
        for j in range(dmap.shape[1]):
            if (abs(i-j) <= 1) and same_domain:
                continue
            # LJ
            rij = dmap[i,j]
            sigij = sig[i,j]
            lamij = lam[i,j]

            u_ah[i,j] = ah_potential(rij,sigij,eps_lj,lamij,rc_lj)

            # YU
            if rij <= rc_yu:
                u_yu[i,j] = yukawa_potential(rij,qmap[i,j],k_yu,rc_yu=rc_yu)
            else:
                u_yu[i,j] = 0.
    return u_ah, u_yu

@nb.jit(nopython=True)
def yukawa_potential(
    r: float, q: float, kappa_yu: float, rc_yu: float = 4.0
) -> float:
    """Return the shifted Yukawa energy at separation ``r``."""
    # q = epsi_yu * epsj_yu
    shift = np.exp(-kappa_yu*rc_yu)/rc_yu
    u = q * (np.exp(-kappa_yu*r)/r - shift)
    return cast(float, u)

@nb.jit(nopython=True)
def lj_potential(r: float, sig: float, eps: float) -> float:
    """Return the Lennard-Jones energy at separation ``r``."""
    ulj = 4.*eps*((sig/r)**12 - (sig/r)**6)
    return ulj

@nb.jit(nopython=True)
def ah_potential(r: float, sig: float, eps: float, l: float, rc: float) -> float:
    """Return the shifted, cutoff Ashbaugh-Hatch energy at separation ``r``."""
    if r <= 2**(1./6.)*sig:
        ah = lj_potential(r,sig,eps) - l * lj_potential(rc,sig,eps) + eps * (1 - l)
    elif r <= rc:
        ah = l * (lj_potential(r,sig,eps) - lj_potential(rc,sig,eps))
    else:
        ah = 0.
    return ah

def calc_dmap(domain0: Any, domain1: Any) -> FloatArray:
    """Return the periodic pairwise distance map between atom groups in nm.

    The result has shape ``(len(domain0), len(domain1))``.
    """
    dmap = distances.distance_array(domain0.positions, # reference
                                    domain1.positions, # configuration
                                    box=domain0.dimensions) / 10.
    return cast(FloatArray, dmap)


def calc_raw_dmap(pos0: FloatArray, pos1: FloatArray) -> FloatArray:
    """Return pairwise distances in the coordinate arrays' input units."""
    dmap = distances.distance_array(pos0,pos1)
    return cast(FloatArray, dmap)


def self_distances(
    pos: FloatArray, box: FloatArray | None = None
) -> FloatArray:
    """Return a symmetric self-distance matrix for a coordinate array.

    When ``box`` is supplied, distances use the minimum-image convention. The
    result has shape ``(len(pos), len(pos))`` and a zero diagonal.
    """
    N = len(pos)
    dmap = np.zeros((N,N))
    if box is not None:
        d = distances.self_distance_array(pos,box)
    else:
        d = distances.self_distance_array(pos)
    k = 0
    for i in range(N):
        for j in range(i + 1, N):
            dmap[i, j] = d[k]
            dmap[j, i] = d[k]
            k += 1
    return dmap

def calc_wcn(
    comp: Any,
    pos: FloatArray,
    fdomains: InputPath | None = None,
    ssonly: bool = True,
    r0: float = 0.7,
) -> FloatArray:
    """Calculate the weighted contact number for every bead.

    ``pos`` and switching distance ``r0`` are in nm. The returned array contains
    one contact number per bead; ``ssonly`` restricts pairs to shared domains.
    """
    N = len(pos)
    # print(f'N: {N}')
    dmap = calc_raw_dmap(pos,pos)
    # dmap = self_distances(pos)

    if ssonly:
        ssdomains = get_ssdomains(comp.name, cast(InputPath, fdomains))
        wcn = np.zeros((N))
        for i in range(N-1):
            for j in range(i+1,N):
                ss = False
                if fdomains is not None:
                    for ssdom in ssdomains:
                        if (i in ssdom) and (j in ssdom):
                            ss = True
                if ss:
                    # print('adding wcn')
                    wcn[i] += (1 - (dmap[i,j]/r0)**6) / (1 - (dmap[i,j]/r0)**12)
    else:
        wcn = (1 - (dmap/r0)**6) / (1 - (dmap/r0)**12)
        wcn = np.sum(wcn,axis=1) - 1. # subtract self-counting
    return wcn#, wcn_binary

# def calc_cmap(domain0,domain1,cutoff=1.5):
#     """ Contact map for single configuration

#     Input: Atom groups
#     Output: Contact map
#     """
#     # Cutoff in nm
#     dmap = calc_dmap(domain0,domain1)
#     cmap = np.where(dmap<cutoff,1,0)
#     return(cmap)

def calc_cmap(domain0: Any, domain1: Any, cutoff: float = 1.0) -> FloatArray:
     """Return a smooth contact map between two MDAnalysis atom groups.

     The result has shape ``(len(domain0), len(domain1))`` with values from zero
     to one; ``cutoff`` is expressed in nm.
     """
     # Cutoff in nm
     dmap = calc_dmap(domain0,domain1)
     cmap = .5 - .5*np.tanh((dmap-cutoff)/.3)
     return(cmap)

def cmap_traj(
    u: Any,
    domain0: Any,
    domain1: Any,
    cutoff: float = 1.0,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
) -> FloatArray:
    """Return the trajectory-averaged smooth contact map.

    The array has shape ``(len(domain0), len(domain1))`` and contains mean
    contact weights over the selected frames.
    """
    cmap = np.zeros((len(domain0),len(domain1)))
    for ts in u.trajectory[start:end:step]:
        cmap += calc_cmap(domain0,domain1,cutoff)
    cmap /= len(u.trajectory[start:end:step])
    return cmap

def calc_fnc(
    u: Any,
    uref: Any,
    selstr: str,
    cutoff: float = 1.5,
    kmax: int = 1,
    bfac: Sequence[float] = (),
    sig_shift: float = 0.8,
    width: float = 50.0,
) -> FloatArray:
    """Calculate the fraction of native contacts for each trajectory frame.

    Native contacts come from ``uref`` after excluding diagonals through
    ``kmax``. The result contains one normalized contact fraction per frame.
    """
    agref = uref.select_atoms(selstr)
    ag = u.select_atoms(selstr)

    if len(bfac) > 0:
        x0 = agref.indices[0]
        x1 = agref.indices[-1]+1
        bfac = bfac[x0:x1]
        bfacmat = np.add.outer(bfac,bfac) / 2.
        sigmoid = np.exp(width*(bfacmat-sig_shift)) / (np.exp(width*(bfacmat-sig_shift)) + 1.)
    else:
        sigmoid = 1.
    fnc = np.zeros((len(u.trajectory)))
    cref = calc_cmap(agref,agref,cutoff=cutoff)
    for k in range(-kmax,kmax+1): # kmax: diagonals to exclude (up to kmax bonds apart)
        cref -= np.diag(np.diag(cref,k=k),k=k) # delete trivial contacts (self and bonded)
    # cref *= sigmoid
    cref_sum = np.sum(cref*sigmoid)
    print('# native contacts:', cref_sum/2.)
    for t,ts in enumerate(u.trajectory):
        c = calc_cmap(ag,ag,cutoff=cutoff)
        # c *= sigmoid
        cnat = c*cref
        cnat_sum = np.sum(cnat*sigmoid)
        fnc[t] = cnat_sum/cref_sum
    return fnc

def calc_rmsd(
    u: Any,
    uref: Any,
    select: str = "all",
    f_out: InputPath | None = None,
    step: int = 1,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Calculate RMSD to reference and mean structures plus per-atom RMSF.

    Returns ``(reference_rmsd, mean_rmsd, mean_rmsf)``. RMSD values use the
    transposed MDAnalysis result-table layout and distances are in Å. When
    ``f_out`` is supplied, the final mean structure is written there.
    """
    # print('First alignment')
    _ = AlignTraj(u, uref, select=select, in_memory=True).run(step=step) # align to crystal structure
    Rref = rms.RMSD(u,uref,select=select) # get RMSD to reference
    Rref.run(step=step)
    coords = u.trajectory.timeseries(u.atoms,step=step)
    coords_mean = coords.mean(axis=1) # get mean structure
    u_mean = Merge(u.atoms) # new universe from mean structure
    u_mean.load_new(coords_mean[:, None, :], order="afc")

    # print('Second alignment')
    _ = AlignTraj(u, u_mean, select=select, in_memory=True).run(step=step) # align to mean structure
    coords = u.trajectory.timeseries(u.atoms,step=step) # get coords
    coords_mean = coords.mean(axis=1) # get new mean
    u_mean2 = Merge(u.atoms)
    u_mean2.load_new(coords_mean[:, None, :], order="afc")

    Rmean = rms.RMSD(u,u_mean2,select=select) # get RMSD to new mean structure
    Rmean.run(step=step)

    sel = u.select_atoms(select)
    RMSFmean = rms.RMSF(sel).run(step=step)

    if f_out is not None:
        u_mean2.select_atoms(select).write(f_out)
    return Rref.results.rmsd.T,Rmean.results.rmsd.T,RMSFmean.results.rmsf

def get_masses(
    seq: Sequence[str], residues: pd.DataFrame, charge_termini: bool = True
) -> FloatArray:
    """Return residue masses in Da, optionally including terminal atoms."""
    lseq = list(seq)
    masses = np.array(residues.loc[lseq,'MW'].values, dtype=np.float64)
    if charge_termini:
        masses[0] += 2.
        masses[-1] += 16.
    return masses

def calc_rg(
    u: Any,
    ag: Any,
    seq: Sequence[str] = (),
    residues: pd.DataFrame | None = None,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> FloatArray:
    """Return the radius of gyration in nm for each selected frame.

    Sequence-derived masses are used when ``seq`` and ``residues`` are supplied;
    otherwise all atoms receive equal weight.
    """
    if len(seq) > 0:
        masses = get_masses(seq, cast(pd.DataFrame, residues))
        # print(masses)
    else:
        masses = np.array([1. for _ in range(len(ag.atoms))])

    rogs: list[float] = []
    for t, ts in enumerate(u.trajectory[start:stop:step]):
        com = ag.center(weights=masses)
        pos = (ag.positions - com) / 10.
        rog_sq = np.einsum('i,i->',masses,np.einsum('ij,ij->i',pos,pos))/np.sum(masses)
        rog = np.sqrt(rog_sq)
        rogs.append(cast(float, rog))
    return np.array(rogs)

def calc_ete(
    u: Any,
    ag: Any,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> tuple[FloatArray, float, float]:
    """Calculate end-to-end distances across a trajectory.

    Returns ``(distances, mean, standard_error)`` in nm for selected frames.
    """
    ete_values: list[float] = []
    # etes2 = []
    for t, ts in enumerate(u.trajectory[start:stop:step]):
        ete = np.linalg.norm(ag[0].position-ag[-1].position) / 10.
        ete_values.append(cast(float, ete))
        # etes2.append(ete**2)
    etes = np.array(ete_values)
    ete_m = np.mean(etes)
    ete_sem = sem(etes)
    # ete2_m = np.mean(etes2)
    # etes = np.array(etes)
    return etes, cast(float, ete_m), cast(float, ete_sem)  # , ete2_m

def calc_ocf(
    u: Any,
    ag: Any,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Calculate orientational correlation versus separation along a chain.

    Returns ``(mean_ocf, sem_ocf)`` arrays indexed by bond separation.
    """
    ocf_values: list[list[float]] = []
    for t,ts in enumerate(u.trajectory[start:stop:step]):
        x = ag.positions / 10.
        xb = x[1:] - x[:-1]
        Lxb = np.linalg.norm(xb,axis=1)
        xbred = (xb.T / Lxb).T
        dots: list[list[float]] = [[] for _ in range(len(xbred))]

        for idx0,xb0 in enumerate(xbred,start=0):
            for idx1,xb1 in enumerate(xbred[idx0:],start=idx0):
                dot = np.dot(xb0,xb1)
                ij = idx1-idx0
                dots[ij].append(cast(float, dot))
        dots_avg: list[float] = []
        for dot in dots:
            dots_avg.append(cast(float, np.mean(dot)))
        ocf_values.append(dots_avg)
    ocfs = np.array(ocf_values)
    ocf = np.mean(ocfs,axis=0)
    ocf_sem = sem(ocfs)
    return cast(FloatArray, ocf), cast(FloatArray, ocf_sem)

#### SCALING EXPONENT

def scaling_exp(n: FloatArray, r0: float, v: float) -> FloatArray:
    """Evaluate the polymer scaling relation ``r0 * n**v``."""
    rh = r0 * n**v
    return rh

def fit_scaling_exp(
    u: Any,
    ag: Any,
    r0: float | None = None,
    traj: bool = True,
    start: int | None = None,
    stop: int | None = None,
    step: int | None = None,
    slic: Sequence[int] = (),
    ij0: int = 5,
) -> tuple[IntArray, FloatArray, float, float, float]:
    """Fit the internal-distance scaling exponent of a single chain.

    Returns ``(sequence_separation, rms_distance, r0, nu, nu_error)``. The first
    two entries are arrays indexed by residue separation, distances and ``r0``
    are in nm, and ``nu_error`` is obtained from the fit covariance.
    """
    N = len(ag)
    dmap = np.zeros((N,N))
    if traj:
        if len(slic) == 0:
            for t,ts in enumerate(u.trajectory[start:stop:step]):
                m = calc_dmap(ag,ag)
                dmap += m**2 # in nm
            dmap /= len(u.trajectory[start:stop:step])
        else:
            for t,ts in enumerate(u.trajectory[slic]):
                m = calc_dmap(ag,ag)
                dmap += m**2 # in nm
            dmap /= len(u.trajectory[slic])
        dmap = np.sqrt(dmap) # RMS
    else:
        dmap = calc_dmap(ag,ag) # in nm
    ij = np.arange(N)
    dij_values: list[list[float]] = [[] for _ in range(N)]
    for i_value in ij:
        i = int(i_value)
        for j in range(i,N):
            dij_values[j-i].append(float(dmap[i,j])) # in nm

    dij = np.array([np.mean(values) for values in dij_values])
    # print(ij.shape)
    # print(dij.shape)
    if r0 is None:
        (r0, v), pcov = curve_fit(scaling_exp,ij[ij>ij0],dij[ij>ij0])
        perr = np.sqrt(np.diag(pcov))
        verr = perr[1]
        # print(pcov)
    else:
        v, pcov = curve_fit(lambda x, v: scaling_exp(x,r0,v), ij[ij>ij0], dij[ij>ij0])
        v = v[0]
        perr = np.sqrt(np.diag(pcov))
        verr = perr[0]
    return (
        cast(IntArray, ij),
        cast(FloatArray, dij),
        cast(float, r0),
        cast(float, v),
        cast(float, verr),
    )

def save_conf_prop(
    path: InputPath,
    name: str,
    residues_file: InputPath,
    output_path: InputPath,
    start: int = 0,
    is_idr: bool = True,
    select: str = "all",
    cutoff: float = 1.0,
    kmax: int = 3,
) -> None:
    """Calculate and save single-chain conformational observables.

    Writes per-frame ``rgs.npy`` and ``rees.npy``, a ``conf_prop.csv`` summary,
    and ``cmap.npy``. For IDRs it additionally writes
    ``internal_distances.npy`` and reports the fitted scaling exponent.
    """
    path = os.fspath(path)
    output_path = os.fspath(output_path)
    residues = pd.read_csv(residues_file).set_index('three')
    u = mda.Universe(f'{path:s}/top.pdb',f'{path:s}/{name:s}.dcd',in_memory=True)
    ag = u.select_atoms(select)
    rgs = calc_rg(u,ag,ag.resnames.tolist(),residues,start=start)
    np.save(output_path+'/rgs.npy',rgs)
    block_rg = BlockAnalysis(rgs)
    block_rg.SEM()
    rees, _, _ = calc_ete(u,ag,start=start)
    np.save(output_path+'/rees.npy',rees)
    block_ree = BlockAnalysis(rees)
    block_ree.SEM()
    df_analysis = pd.DataFrame(index=['Rg','Ree'],columns=['value','error'])
    df_analysis.loc['Rg','value'] = np.mean(rgs)
    df_analysis.loc['Rg','error'] = block_rg.sem
    df_analysis.loc['Ree','value'] = np.mean(rees)
    df_analysis.loc['Ree','error'] = block_ree.sem
    if is_idr:
        ij, dij, _, nu, nu_err = fit_scaling_exp(u,ag,start=start)
        df_analysis.loc['nu','value'] = nu
        df_analysis.loc['nu','error'] = nu_err
        np.save(output_path+'/internal_distances.npy',[ij,dij])
    df_analysis.to_csv(output_path+'/conf_prop.csv')
    ag = u.select_atoms(select)
    cmap = cmap_traj(u,ag,ag,cutoff,start)
    for k in range(-kmax,kmax+1): # kmax: diagonals to exclude (up to kmax bonds apart)
         cmap -= np.diag(np.diag(cmap,k=k),k=k)
    np.save(output_path+'/cmap.npy',cmap)

class SlabAnalysis:
    """Analyze concentration and structural profiles across a slab simulation.

    The workflow centers a trajectory, builds reference and optional client
    concentration profiles along z, identifies dense and dilute regions, and
    writes concentrations, blocking errors, and transfer free energies. Extra
    methods produce orientation, radius-of-gyration, composition, and
    center-of-mass profiles. Distances used internally for histogramming are in
    Å, while public profile coordinates and concentrations are in nm and mM.
    """

    def __init__(
        self,
        name: str,
        input_path: InputPath = ".",
        output_path: InputPath = ".",
        input_pdb: InputPath = "top.pdb",
        input_dcd: InputPath | None = None,
        centered_dcd: InputPath = "traj.dcd",
        ref_chains: tuple[int, int] | None = None,
        ref_name: str | None = None,
        client_chain_list: Sequence[tuple[int, int]] = (),
        client_names: Sequence[str] = (),
        verbose: bool = False,
    ) -> None:
        """Configure slab inputs, component chain ranges, and output paths."""
        self.name = name
        self.input_path = os.fspath(input_path)
        self.output_path = os.fspath(output_path)
        self.input_pdb = os.fspath(input_pdb)
        if input_dcd is None:
            input_dcd = f'{self.name}.dcd'
        self.input_dcd = os.fspath(input_dcd)
        self.centered_dcd = os.fspath(centered_dcd)
        self.ref_chains = ref_chains
        self.ref_name = ref_name
        if self.ref_name is None:
            self.ref_name = 'ref'
        self.client_chain_list = list(client_chain_list)
        self.client_names = list(client_names)
        if len(self.client_names) == 0:
            self.client_names = [f'client_{idx}' for idx in range(len(self.client_chain_list))]
        self.verbose = verbose

        u = mda.Universe(f'{self.input_path}/{self.input_pdb}')
        self.lz, self.edges, self.z = self.calc_z_Angstr(u)
        _, self.edges_nm, self.z_nm = self.calc_z_nm_centered(u)

        self.n_bins = len(self.z)
        os.system(f'mkdir -p {self.output_path}')

        if self.verbose:
            print(f'Input pdb: {self.input_path}/{self.input_pdb}')
            print(f'Input dcd: {self.input_path}/{self.input_dcd}')

    def center(
        self,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
        center_target: str = "ref",
    ) -> None:
        """Center and unwrap a slab trajectory around reference or all atoms.

        Selected frames are written to ``centered_dcd``; no value is returned.
        """

        u = mda.Universe(f'{self.input_path}/{self.input_pdb}', f'{self.input_path}/{self.input_dcd}', in_memory=True)
        n_frames = len(u.trajectory[start:end:step])
        if center_target == 'ref':
            if self.ref_chains is None:
                self.ref_chains = (0, len(u.segments)-1)
            sg_ref = u.segments[self.ref_chains[0]:self.ref_chains[1]+1]
            ag_ref = sg_ref.atoms
            if self.verbose:
                print(f'Using reference chain {self.ref_name} for centering.')
        elif center_target == 'all':
            ag_ref = u.atoms
            if self.verbose:
                print('Using all chains for centering.')
        else:
            raise
        ag = u.atoms
        n_atoms = ag.n_atoms
        # create list of bonds
        bonds: list[tuple[int, int]] = []
        assert u.segments is not None
        for segment in u.segments:
            for i in segment.atoms.indices[:-1]:
                bonds.extend([(i, i+1)])
        u.add_TopologyAttr('bonds', bonds)

        # hs = np.zeros((n_frames,n_bins))
        with mda.Writer(f'{self.input_path}/{self.centered_dcd}', n_atoms) as W:
            for t,ts in tqdm(enumerate(u.trajectory[start:end:step]),total=n_frames):
                # shift max density to center
                zpos = ag_ref.positions.T[2]
                h, e = np.histogram(zpos,bins=self.edges)
                zmax = self.z[np.argmax(h)]
                ag.translate(np.array([0,0,-zmax+0.5*self.lz]))
                # wrap
                ts = transformations.wrap(ag)(ts)
                # shift weighted average of slab density to center
                zpos = ag_ref.positions.T[2]
                h, e = np.histogram(zpos, bins=self.edges)
                zpatch, hpatch = self.calc_zpatch(self.z,h)
                zmid = np.average(zpatch,weights=hpatch)
                ag.translate(np.array([0,0,-zmid+0.5*self.lz]))
                # wrap
                ts = transformations.wrap(ag)(ts)
                # zpos = ag_ref.positions.T[2]
                # h, e = np.histogram(zpos,bins=self.edges)
                # hs[t] = h
                # make chains whole for trajectory output
                ts = transformations.unwrap(ag)(ts)
                W.write(ag)
        if self.verbose:
            print(f'Written {n_frames} centered frames to {self.input_path}/{self.centered_dcd}')

    def calc_profiles(
        self,
        start: int | None = None,
        end: int | None = None,
        step: int = 1,
        save_individual_profiles: bool = True,
    ) -> None:
        """Calculate z concentration profiles for reference and client chains.

        Individual outputs have shape ``(n_frames, n_bins)`` in mM. The combined
        ``<name>_profiles.npy`` stores z coordinates in its first row followed by
        trajectory-averaged profiles for the reference and each client.
        """

        self.load_traj(centered=True, step=step)
        self.load_ref()

        n_frames = len(self.u.trajectory[start:end:step])
        binwidth = 1 # 0.1 nm
        volume = self.u.dimensions[0]*self.u.dimensions[1]*binwidth/1e3 # volume of a slice in nm3
        conv_ref = 10/6.02214/self.nbeads_ref/volume*1e3 # conversion to mM

        # Reference profile
        h_ref = np.zeros((n_frames,self.n_bins))
        for t,ts in enumerate(self.u.trajectory[start:end:step]):
            ts = transformations.wrap(self.ag_ref)(ts)
            zpos = self.ag_ref.positions.T[2]
            h, e = np.histogram(zpos,bins=self.edges)
            h_ref[t] = h * conv_ref # mM
        if save_individual_profiles:
            np.save(f'{self.output_path}/{self.name}_{self.ref_name}_profile.npy', h_ref) # in mM

        h_ref_mean = h_ref.mean(axis=0) # mM
        all_profiles = [self.z_nm, h_ref_mean]

        # Client profiles
        for i, (first,last) in enumerate(self.client_chain_list):
            sg_sel = self.u.segments[first:last+1]
            ag_sel = sg_sel.atoms
            nbeads_sel = len(sg_sel[0].atoms)
            if self.verbose:
                print(f'Client {i}: name {self.client_names[i]}; chains {first}-{last}; nbeads: {nbeads_sel}')
            # print(nbeads_sel)
            conv_sel = 10/6.02214/nbeads_sel/volume*1e3 # conversion to mM

            h_sel = np.zeros((n_frames,self.n_bins))
            for t,ts in enumerate(self.u.trajectory[start:end:step]):
                # wrap for density profile calculation
                ts = transformations.wrap(ag_sel)(ts)
                zpos = ag_sel.positions.T[2]
                h, e = np.histogram(zpos,bins=self.edges)
                h_sel[t] = h * conv_sel

            if save_individual_profiles:
                np.save(f'{self.output_path}/{self.name}_{self.client_names[i]}_profile.npy', h_sel) # individual profiles
            h_sel_mean = h_sel.mean(axis=0) # mM
            all_profiles.append(h_sel_mean)
        self.all_profiles = np.array(all_profiles)

        np.save(f'{self.output_path}/{self.name}_profiles.npy', self.all_profiles) # all trajectory-averaged profiles
        if self.verbose:
            print(f'Output written to {self.output_path}/')

    def calc_concentrations(
        self,
        pden: float = 2.0,
        pdil: float = 8.0,
        dGmin: float = -10.0,
        write_conc_arrays: bool = True,
    ) -> None:
        """Calculate dense/dilute concentrations and transfer free energies.

        Results for every component are written to ``<name>_ps_results.csv``;
        optional per-frame dense and dilute concentration arrays are saved as
        NumPy files. This method returns nothing.
        """

        self.pden, self.pdil = pden, pdil
        self.dGmin = dGmin
        self.df_results = pd.DataFrame(dtype=object)
        self.write_conc_arrays = write_conc_arrays

        # Reference concentrations
        if self.ref_chains is None:
            u = mda.Universe(f'{self.input_path}/'+self.input_pdb)
            self.ref_chains = (0, len(u.segments)-1)

        h = np.load(f'{self.output_path}/{self.name}_{self.ref_name}_profile.npy')
        results = self.calc_single_conc(h,ref=True)
        results['first_chain'], results['last_chain'] = self.ref_chains[0], self.ref_chains[1]
        self.save_conc_results(f'{self.name}_{self.ref_name}', results)

        # Client concentrations
        for i, (first,last) in enumerate(self.client_chain_list):
            h = np.load(f'{self.output_path}/{self.name}_{self.client_names[i]}_profile.npy')
            results = self.calc_single_conc(h,ref=False)
            results['first_chain'], results['last_chain'] = first, last
            self.save_conc_results(f'{self.name}_{self.client_names[i]}', results)

        self.df_results.to_csv(f'{self.output_path}/{self.name}_ps_results.csv')

    def save_conc_results(
        self, comp_name: str, results: ConcentrationResults
    ) -> None:
        """Append scalar concentration results and optionally save frame arrays."""
        for key, val in results.items():
            if key in ['dense_array', 'dilute_array']:
                if self.write_conc_arrays:
                    np.save(f'{self.output_path}/{comp_name}_{key}.npy', val)
            else:
                self.df_results.loc[comp_name, key] = val

    def calc_single_conc(
        self, h: FloatArray, ref: bool = True
    ) -> ConcentrationResults:
        """Calculate phase concentrations and errors from frame profiles.

        ``h`` has shape ``(n_frames, n_bins)`` in mM. The result maps cutoff
        positions, scalar dense/dilute concentrations and errors, per-frame
        concentration arrays, and ``dG``/``dG_err`` in units of kT. Reference
        profiles also establish the cutoffs reused for client profiles.
        """

        hm = np.mean(h,axis=0)

        if ref:
            self.cutoffs_dense, self.cutoffs_dilute = self.fit_profile(self.z_nm, hm, self.pden, self.pdil)

        results: ConcentrationResults = {}
        results['cutoffs_dense_right'], results['cutoffs_dense_left'] = self.cutoffs_dense[0], self.cutoffs_dense[1]
        results['cutoffs_dilute_right'], results['cutoffs_dilute_left'] = self.cutoffs_dilute[0], self.cutoffs_dilute[1]

        bool_dense = np.logical_and(self.z_nm<self.cutoffs_dense[0],self.z_nm>self.cutoffs_dense[1]) # dense
        bool_dilute = np.logical_or(self.z_nm>self.cutoffs_dilute[0],self.z_nm<self.cutoffs_dilute[1]) # dilute

        cden = hm[bool_dense].mean() # average dense concentration
        cdil = hm[bool_dilute].mean() # average dilute concentration

        results['c_dilute'], results['c_dense'] = cdil, cden # mM

        denarray = np.apply_along_axis(lambda a: a[bool_dense].mean(), 1, h)
        dilarray = np.apply_along_axis(lambda a: a[bool_dilute].mean(), 1, h) # concentration in range [bool_dilute]

        results['dense_array'], results['dilute_array'] = denarray, dilarray

        eden, edil = self.calc_block_errors(denarray, dilarray)

        results['c_dilute_err'], results['c_dense_err'] = edil, eden # mM

        dG, dG_error = self.calc_dG(cdil,edil,cden,eden,ndraws=10000,dGmin=self.dGmin)
        results['dG'], results['dG_err'] = dG, dG_error # kT
        return results

    def load_traj(self, centered: bool = False, step: int = 1) -> None:
        """Load the original or centered trajectory into ``self.u``."""
        if centered:
            dcd = self.centered_dcd
            traj_str = 'centered'
        else:
            dcd = self.input_dcd
            traj_str = 'original'
        self.u = mda.Universe(f'{self.input_path}/{self.input_pdb}', f'{self.input_path}/{dcd}', in_memory=True)
        if self.verbose:
            print(f'Loaded {traj_str} trajectory {self.input_path}/{dcd}')
            print(f'nframes: {len(self.u.trajectory[::step])}')

    def load_ref(self) -> None:
        """Select reference chains and cache their atom groups and bead count."""
        if self.ref_chains is None:
            self.ref_chains = (0, len(self.u.segments)-1)
        self.sg_ref = self.u.segments[self.ref_chains[0]:self.ref_chains[1]+1]
        self.ag_ref_per_chain = [seg.atoms for seg in self.sg_ref] # all beads per chain
        self.ag_ref = self.sg_ref.atoms
        self.nbeads_ref = len(self.sg_ref[0].atoms)
        if self.verbose:
            print(f'Reference: name {self.ref_name}; chains {self.ref_chains[0]}-{self.ref_chains[1]}; nbeads: {self.nbeads_ref}')

    @staticmethod
    @nb.jit(nopython=True)
    def distribute_monomers(
        prop: float,
        prop_binned: FloatArray,
        bin_counts: FloatArray,
        bead_positions: FloatArray,
        L: float,
    ) -> tuple[FloatArray, FloatArray]:
        """Accumulate a scalar property and sample counts in periodic z bins.

        Returns the modified ``(prop_binned, bin_counts)`` arrays.
        """
        for bpos in bead_positions:
            while (bpos >= L) or (bpos < 0.):
                bpos -= (bpos // L) * L
            bin_idx = int(bpos)
            prop_binned[bin_idx] += prop
            bin_counts[bin_idx] += 1
        return prop_binned, bin_counts

    def calc_orientations(self, step: int = 1) -> None:
        """Calculate the reference-chain orientational order profile along z.

        Writes one mean order parameter per Å-wide bin to ``<name>_sz.npy``.
        """
        
        self.load_traj(centered=True, step=step)
        self.load_ref()

        z = np.array([0.,0.,1.])

        bin_counts = np.zeros((int(self.lz)))
        sz_binnned = np.zeros((int(self.lz)))

        for idx, seg in tqdm(enumerate(self.ag_ref_per_chain),total=len(self.ag_ref_per_chain)):
            for t, ts in enumerate(self.u.trajectory[::step]):
                a = seg.principal_axes()[2]
                cos = self.calc_cos(a,z)
                sz = 3./2.*cos**2 - 1./2.
                bead_positions = seg.positions[:,2]
                sz_binnned, bin_counts = self.distribute_monomers(sz, sz_binnned, bin_counts, bead_positions, self.lz)
        
        sz_m = np.zeros((int(self.lz)))
        for bin_idx, sz in enumerate(sz_binnned):
            if bin_counts[bin_idx] == 0:
                sz_m[bin_idx] = 0.
            else:
                sz_m[bin_idx] = sz / bin_counts[bin_idx]
        
        np.save(f'{self.output_path}/{self.name}_sz.npy',sz_m)

    def calc_rgs(self, step: int = 1) -> None:
        """Calculate the reference-chain radius-of-gyration profile along z.

        Writes one root-mean-square radius in nm per Å-wide bin to
        ``<name>_rg.npy``; empty bins contain NaN.
        """

        self.load_traj(centered=True, step=step)
        self.load_ref()

        bin_counts = np.zeros((int(self.lz)))
        rg2_binned = np.zeros((int(self.lz)))

        for idx, seg in tqdm(enumerate(self.ag_ref_per_chain),total=len(self.ag_ref_per_chain)):
            for t,ts in enumerate(self.u.trajectory[::step]):
                rg2 = (seg.radius_of_gyration() / 10.)**2 # nm
                bead_positions = seg.positions[:,2]
                rg2_binned, bin_counts = self.distribute_monomers(rg2, rg2_binned, bin_counts, bead_positions, self.lz)

        rg_m = np.zeros((int(self.lz)))

        for bin_idx, rg2 in enumerate(rg2_binned):
            if bin_counts[bin_idx] == 0:
                rg_m[bin_idx] = np.nan
            else:
                rg_m[bin_idx] = np.sqrt(rg2 / bin_counts[bin_idx])
        np.save(f'{self.output_path}/{self.name}_rg.npy',rg_m)

    def plot_density_profiles(self) -> None:
        """Plot mean concentration profiles and phase cutoffs to a PDF."""
        fig, ax = plt.subplots(figsize=(8,4))

        for c1,c2 in zip(self.cutoffs_dense,self.cutoffs_dilute):
            ax.axvline(c1,color='gray', ls='dashed')
            ax.axvline(c2,color='gray', ls='dotted')

        profiles = np.load(f'{self.output_path}/{self.name}_profiles.npy') # all trajectory-averaged profiles
        z = profiles[0]
        h_ref = profiles[1]

        ax.plot(z, h_ref, color='black', label=self.ref_name) # reference

        if len(profiles) > 2:
            for idx, h in enumerate(profiles[2:]):
                ax.plot(z, h, label=f'{self.client_names[idx]}') # Clients

        ax.set(xlabel='z [nm]', ylabel='Concentration [mM]')
        ax.set(yscale='log')
        ax.set(title=self.name)
        ax.legend(fontsize=8)

        fig.tight_layout()
        fig.savefig(f'{self.output_path}/{self.name}_profiles.pdf')

    def calc_com_traj(
        self,
        residues_file: InputPath,
        step: int = 1,
        index_col: str = "three",
    ) -> None:
        """Write the reference-chain center-of-mass trajectory.

        Produces ``<name>_com_top.pdb`` and ``<name>_com_traj.dcd`` with one
        center-of-mass particle per reference chain.
        """

        self.load_traj(centered=True, step=step)
        self.load_ref()

        print(self.ref_chains)

        residues = pd.read_csv(residues_file, index_col=index_col)

        traj = md.load_dcd(
            f'{self.input_path}/traj.dcd',
            top=f'{self.input_path}/{self.input_pdb}')

        chain_prop: dict[str, dict[str, Any]] = {}
        chain_name = cast(str, self.ref_name)
        n_chains = 0
        chainids = cast(tuple[int, int], self.ref_chains)

        chain_prop[chain_name] = {}
        # if type(chainids) is int:
        #     chainids = (chainids, chainids)
        seq = [res.name for res in traj.top.chain(chainids[0]).residues]
        if len(seq[0]) == 1:
            seq = [SeqUtils.seq3(res).upper() for res in seq]  # type: ignore[no-untyped-call]
        mws = residues.loc[seq,'MW'].values
        mws[0] += 2
        mws[-1] += 16
        print(mws)
        chain_prop[chain_name]['ids'] = np.arange(chainids[0],chainids[1]+1)
        n_chains += chain_prop[chain_name]['ids'].size
        chain_prop[chain_name]['N'] = len(seq)
        chain_prop[chain_name]['MWs'] = mws
        chain_prop[chain_name]['rgs'] = []

        # calculate traj of chain COM
        cmtop = md.Topology()
        xyz = np.empty((traj.n_frames,n_chains,3))
        for chain_name in chain_prop.keys():
            print(chain_name)
            for chainid in chain_prop[chain_name]['ids']:
                print(chainid)
                chain = traj.top.chain(chainid)
                mws = chain_prop[chain_name]['MWs']
                new_chain = cmtop.add_chain()
                res = cmtop.add_residue('COM', new_chain, resSeq=chainid)
                cmtop.add_atom(chain_name, element=traj.top.atom(0).element, residue=res)
                t_chain = traj.atom_slice(traj.top.select(f'chainid {chainid:d}'))
                com = np.sum(t_chain.xyz*mws[np.newaxis,:,np.newaxis],axis=1)/mws.sum()
                xyz[:,new_chain.index,:] = com
        cmtraj = md.Trajectory(xyz, cmtop, traj.time, traj.unitcell_lengths, traj.unitcell_angles)

        # calculate radial distribution function
        cmtraj[0].save_pdb(f'{self.output_path}/{self.name}_com_top.pdb')
        cmtraj.save_dcd(f'{self.output_path}/{self.name}_com_traj.dcd')

    def calc_aa_bins(self, step: int = 1) -> None:
        """Count each amino-acid type by z bin and save ``<name>_aa_bins.npy``."""

        aminoacids = "ACDEFGHIKLMNPQRSTVWY"

        self.load_traj(centered=True, step=step)
        self.load_ref()

        self.bins = np.zeros((int(self.lz), 20))

        if len(self.ag_ref_per_chain[0].names[0]) > 1: # three letter res
            bead_names = [
                str(SeqUtils.seq1(s))  # type: ignore[no-untyped-call]
                for s in self.ag_ref_per_chain[0].names
            ]
        else:
            bead_names = [str(s) for s in self.ag_ref_per_chain[0].names]

        aa_indices = tuple(int(aminoacids.index(aa)) for aa in bead_names)
        # print(aa_indices)

        for t, ts in tqdm(enumerate(self.u.trajectory[::step]), total=len(self.u.trajectory[::step])): 
            for seg in self.ag_ref_per_chain:
                bead_positions = seg.positions[:,2]
                self.bins = self.aa_into_bins(self.bins, bead_positions, aa_indices, self.lz)
        np.save(f'{self.output_path}/{self.name}_aa_bins.npy', self.bins)

    @staticmethod
    @nb.jit(nopython=True)
    def aa_into_bins(
        bins: FloatArray,
        bead_positions: FloatArray,
        aa_indices: tuple[int, ...],
        L: float,
    ) -> FloatArray:
        """Accumulate amino-acid counts into periodic z bins and return them."""
        for resid, bpos in enumerate(bead_positions):
        # for bpos, aa_idx in zip(bead_positions, aa_indices):
            aa_idx = aa_indices[resid]
            # bpos = self.wrap_bead(bpos, L)
            while (bpos >=  L) or (bpos < 0.):
                bpos -= (bpos // L) *  L
            bin_idx = int(bpos)
            # aa_idx = int(aminoacids.index(aa))
            bins[bin_idx, aa_idx] += 1
        return bins

    def calc_resid_bins(self, step: int = 1) -> None:
        """Count each residue index by z bin and save ``<name>_resid_bins.npy``."""

        self.load_traj(centered=True, step=step)
        self.load_ref()

        nbeads = len(self.sg_ref[0].atoms) # nresidues
        self.bins = np.zeros((int(self.lz), nbeads))

        for t, ts in tqdm(enumerate(self.u.trajectory[::step]), total=len(self.u.trajectory[::step])):
            for seg in self.ag_ref_per_chain:
                bead_positions = seg.positions[:,2]
                self.bins = self.resid_into_bins(self.bins, bead_positions, self.lz)
        np.save(f'{self.output_path}/{self.name}_resid_bins.npy', self.bins)

    @staticmethod
    @nb.jit(nopython=True)
    def resid_into_bins(
        bins: FloatArray, bead_positions: FloatArray, L: float
    ) -> FloatArray:
        """Accumulate residue-index counts into periodic z bins and return them."""
        for resid, bpos in enumerate(bead_positions):
            while (bpos >=  L) or (bpos < 0.):
                bpos -= (bpos // L) *  L
            # bpos = self.wrap_bead(bpos, L)
            bin_idx = int(bpos)
            bins[bin_idx, resid] += 1
        return bins

    # @staticmethod
    # @nb.jit(nopython=True)
    # def wrap_bead(bpos, L):
    #     while (bpos >=  L) or (bpos < 0.):
    #         bpos -= (bpos // L) *  L
    #     return bpos

    @staticmethod
    def calc_cos(a: FloatArray, b: FloatArray) -> float:
        """Return the cosine of the angle between two vectors."""
        cos = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
        return cast(float, cos)

    @staticmethod
    def calc_z_Angstr(u: Any) -> tuple[float, FloatArray, FloatArray]:
        """Return box length, edges, and centers for one-Å z bins."""
        lz = u.dimensions[2]
        edges = np.arange(0,lz+1,1)
        dz = (edges[1] - edges[0]) / 2.
        z = edges[:-1] + dz
        return lz, edges, z

    @staticmethod
    def calc_z_nm_centered(u: Any) -> tuple[float, FloatArray, FloatArray]:
        """Return box length in Å and centered z-bin edges and centers in nm."""
        lz = u.dimensions[2]
        edges = np.arange(-lz/2.,lz/2.+0.0001,1)/10
        dz = (edges[1] - edges[0]) / 2.
        z = edges[:-1] + dz
        return lz, edges, z

    @staticmethod
    def calc_zpatch(
        z: FloatArray, h: NDArray[np.integer[Any]]
    ) -> tuple[FloatArray, FloatArray]:
        """Return coordinates and counts for the largest occupied z patch."""
        cutoff = 0
        ct = 0.
        ct_max = 0.
        zwindow: list[float] = []
        hwindow: list[float] = []
        zpatch: list[float] = []
        hpatch: list[float] = []
        for ix, x in enumerate(h):
            if x > cutoff:
                ct += x
                zwindow.append(z[ix])
                hwindow.append(x)
            else:
                if ct > ct_max:
                    ct_max = ct
                    zpatch = zwindow
                    hpatch = hwindow
                ct = 0.
                zwindow = []
                hwindow = []
        return np.array(zpatch), np.array(hpatch)

    @staticmethod
    def calc_dG(
        c_dil: float,
        e_dil: float,
        c_den: float,
        e_den: float,
        ndraws: int = 10000,
        dGmin: float = -10,
    ) -> tuple[float, float]:
        """Calculate transfer free energy and its Monte Carlo error in kT.

        Returns ``(dG, dG_error)`` for ``log(c_dil / c_den)``. Undefined phases
        yield NaN, and values below ``dGmin`` are clipped to that threshold.
        """
        # Calculate deltaG
        if np.isnan(c_dil) or np.isnan(c_den):
            print("Not converged, setting dG to NaN")
            dG = np.nan
            dG_error = np.nan
        elif c_dil == 0. and c_den > 0.:
            print(f'No dilute phase, setting dG to {dGmin:.1f}')
            dG = dGmin
            dG_error = np.nan
        elif c_den == 0. and c_dil > 0.:
            print(f'No dense phase, setting dG to {-dGmin:.1f}')
            dG = -dGmin
            dG_error = np.nan
        elif c_den == 0. and c_dil == 0.:
            print('No dense or dilute phase, setting dG to NaN')
            dG = np.nan
            dG_error = np.nan
        else:
            dG = np.log(c_dil/c_den)
            spread_dil = np.random.normal(c_dil,e_dil,size=ndraws)
            spread_den = np.random.normal(c_den,e_den,size=ndraws)
            spread_dGs: list[float] = []
            for idraw, (dil,den) in enumerate(zip(spread_dil,spread_den)):
                if dil > 0 and den > 0:
                    spread_dGs.append(np.log(dil/den))
            dG_error = cast(float, np.std(spread_dGs))
        if dG < dGmin:
            dG = dGmin
            dG_error = 0.
            print(f'dG extremely small, setting dG to {dGmin:.1f}')
        return dG, dG_error

    @staticmethod
    def fit_profile(
        z: FloatArray, hm: FloatArray, pden: float, pdil: float
    ) -> tuple[FloatArray, FloatArray]:
        """Fit both slab interfaces and return dense and dilute cutoff pairs.

        Each returned two-element array is ordered ``(right, left)`` and uses
        the same coordinate units as ``z``.
        """
        def profile(
            x: FloatArray, a: float, b: float, c: float, d: float
        ) -> FloatArray:
            """Evaluate the symmetric hyperbolic-tangent interface model."""
            return .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d)

        def residuals(
            params: FloatArray, x_values: FloatArray, h_values: FloatArray
        ) -> FloatArray:
            """Return observed-minus-model profile residuals."""
            return h_values - profile(x_values, *params)
        z1 = z[z>0]
        h1 = hm[z>0]
        z2 = z[z<0]
        h2 = hm[z<0]
        p0=[1,1,1,1]
        res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[100]*4)) # fit to hyperbolic function
        res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[100]*4))

        cutoffs_dense = np.array([res1.x[2]-pden*res1.x[3],-res2.x[2]+pden*res2.x[3]]) # position of interface - half width
        cutoffs_dilute = np.array([res1.x[2]+pdil*res1.x[3],-res2.x[2]-pdil*res2.x[3]]) # get far enough from interface for dilute phase calculation

        return cutoffs_dense, cutoffs_dilute

        if (np.abs(cutoffs_dilute[1]/cutoffs_dilute[0]) > 2) or (np.abs(cutoffs_dilute[1]/cutoffs_dilute[0]) < 0.5): # ratio between right and left should be close to 1
            print('NOT CONVERGED',cutoffs_dense,cutoffs_dilute)
            print(res1.x,res2.x)

    @staticmethod
    def calc_block_errors(
        denarray: FloatArray, dilarray: FloatArray
    ) -> tuple[float, float]:
        """Return blocking errors for dense and dilute concentration arrays."""

        block_den = BlockAnalysis(denarray)
        block_dil = BlockAnalysis(dilarray)

        block_den.SEM()
        block_dil.SEM()

        eden = block_den.sem
        edil = block_dil.sem

        return eden, edil

# # @staticmethod
# @nb.jit(nopython=True)
# def calc_cos(a,b):

#     dot = a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

#     la2 = a[0]**2 + a[1]**2 + a[2]**2
#     lb2 = b[0]**2 + b[1]**2 + b[2]**2

#     cos = dot / math.sqrt(la2 * lb2)
#     # cos = np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))
#     return cos

def calc_com_traj(
    path: InputPath,
    sysname: str,
    output_path: InputPath,
    residues_file: InputPath,
    chainid_dict: dict[str, ChainRange] | None = None,
    start: int | None = None,
    end: int | None = None,
    step: int = 1,
    input_pdb: InputPath = "top.pdb",
    verbose: bool = False,
) -> None:
    """Calculate chain center-of-mass trajectories and radii of gyration.

    ``chainid_dict`` maps component names to a chain ID or inclusive ID range;
    by default all chains belong to ``sysname``. Writes one ``(n_frames,
    n_chains)`` Rg array in nm per component, plus a PDB topology and DCD
    trajectory containing one center-of-mass particle per chain. Returns nothing.
    """
    path = os.fspath(path)
    output_path = os.fspath(output_path)
    input_pdb = os.fspath(input_pdb)
    if chainid_dict is None:
        chainid_dict = {}

    if not os.path.isfile(f'{path:s}/traj.dcd'):
        u = mda.Universe(f'{path:s}/{input_pdb:s}',f'{path:s}/{sysname:s}.dcd',in_memory=True)
        ag = u.select_atoms('all')
        n_atoms = ag.n_atoms
        # create list of bonds
        bonds: list[tuple[int, int]] = []
        for segment in u.segments:
            for i in segment.atoms.indices[:-1]:
                bonds.extend([(i, i+1)])
        u.add_TopologyAttr('bonds', bonds)
        with mda.Writer(f'{path:s}/traj.dcd',n_atoms) as W:
            for t,ts in enumerate(u.trajectory[start:end:step]):
                # make chains whole
                ts = transformations.unwrap(ag)(ts)
                W.write(ag)

    traj = md.load_dcd(f'{path:s}/traj.dcd',top=f'{path:s}/'+input_pdb)
    traj.xyz -= traj.unitcell_lengths[0,:]/2

    if len(chainid_dict) == 0:
        chainid_dict[sysname] = (0, traj.top.n_chains-1)

    residues = pd.read_csv(residues_file, index_col='three')

    chain_prop: dict[str, dict[str, Any]] = {}
    n_chains = 0
    for chain_name, chainids in chainid_dict.items():
        chain_prop[chain_name] = {}
        if isinstance(chainids, int):
            chainids = (chainids, chainids)
        seq = [res.name for res in traj.top.chain(chainids[0]).residues]
        mws = residues.loc[seq,'MW'].values
        mws[0] += 2
        mws[-1] += 16
        chain_prop[chain_name]['ids'] = np.arange(chainids[0],chainids[1]+1)
        n_chains += chain_prop[chain_name]['ids'].size
        chain_prop[chain_name]['N'] = len(seq)
        chain_prop[chain_name]['MWs'] = mws
        chain_prop[chain_name]['rgs'] = []

    # calculate traj of chain COM
    cmtop = md.Topology()
    xyz = np.empty((traj.n_frames,n_chains,3))
    for chain_name in chain_prop.keys():
        if verbose:
            print(chain_name)
        for chainid in chain_prop[chain_name]['ids']:
            if verbose:
                print(chainid)
            chain = traj.top.chain(chainid)
            mws = chain_prop[chain_name]['MWs']
            new_chain = cmtop.add_chain()
            res = cmtop.add_residue('COM', new_chain, resSeq=chainid)
            cmtop.add_atom(chain_name, element=traj.top.atom(0).element, residue=res)
            t_chain = traj.atom_slice(traj.top.select(f'chainid {chainid:d}'))
            com = np.sum(t_chain.xyz*mws[np.newaxis,:,np.newaxis],axis=1)/mws.sum()
            # calculate residue-cm distances
            si = np.linalg.norm(t_chain.xyz - com[:,np.newaxis,:],axis=2)
            # calculate rg
            chain_rg = np.sqrt(np.sum(si**2*mws,axis=1)/mws.sum())
            chain_prop[chain_name]['rgs'].append(chain_rg.tolist())
            xyz[:,new_chain.index,:] = com
    cmtraj = md.Trajectory(xyz, cmtop, traj.time, traj.unitcell_lengths, traj.unitcell_angles)

    for chain_name in chain_prop.keys():
        np.save(output_path+f'/{sysname:s}_{chain_name:s}_rg.npy',np.asarray(chain_prop[chain_name]['rgs']).T)

    # calculate radial distribution function
    cmtraj[0].save_pdb(output_path+f'/{sysname:s}_com_top.pdb')
    cmtraj.save_dcd(output_path+f'/{sysname:s}_com_traj.dcd')

def calc_contact_map(
    path: InputPath,
    sysname: str,
    output_path: InputPath,
    chainid_dict: dict[str, ChainIds] | None = None,
    is_slab: bool = False,
    input_pdb: InputPath = "top.pdb",
) -> None:
    """Calculate and save a residue contact map between component chain sets.

    ``chainid_dict`` maps one or two component names to chain IDs or inclusive
    ranges; one component requests a homotypic map. The output is an
    ``(n_residues_1, n_residues_2)`` array of trajectory-averaged smooth contact
    counts. In slab mode, only the most central reference chain in each frame is
    used and dense/dilute Rg subsets are also saved. Returns nothing.
    """
    path = os.fspath(path)
    output_path = os.fspath(output_path)
    input_pdb = os.fspath(input_pdb)
    if chainid_dict is None:
        chainid_dict = {}

    traj = md.load_dcd(f'{path:s}/traj.dcd',top=f'{path:s}/'+input_pdb)
    traj.xyz -= traj.unitcell_lengths[0,:]/2

    if len(chainid_dict) > 0:
        name_1 = next(iter(chainid_dict))
        chain_1_ids = chainid_dict[name_1]
        if isinstance(chain_1_ids, int):
            chain_1_ids = (chain_1_ids, chain_1_ids)
        if isinstance(chain_1_ids, tuple):
            chainid_dict[name_1] = np.arange(chain_1_ids[0], chain_1_ids[1]+1)
        if len(chainid_dict) > 1:
            name_2 = next(iter(list(chainid_dict.keys())[1:]))
            chain_2_ids = chainid_dict[name_2]
            if isinstance(chain_2_ids, int):
                chain_2_ids = (chain_2_ids, chain_2_ids)
            if isinstance(chain_2_ids, tuple):
                chainid_dict[name_2] = np.arange(chain_2_ids[0], chain_2_ids[1]+1)
        else:
            # if homotypic cmap
            name_2 = name_1
    else:
        name_1 = sysname
        name_2 = name_1
        chainid_dict[name_1] = np.arange(traj.top.n_chains)

    chain_indices = cast(dict[str, IntArray], chainid_dict)

    print(name_1)
    print(chain_indices[name_1])
    print(name_2)
    print(chain_indices[name_2])

    N_res_1 = traj.top.chain(chain_indices[name_1][0]).n_residues
    N_res_2 = traj.top.chain(chain_indices[name_2][0]).n_residues

    if is_slab:
        if not os.path.isfile(output_path+f'/{sysname:s}_ps_results.csv'):
            raise ValueError('Please run functions in SlabAnalysis class first')
        else:
            ps_results = pd.read_csv(output_path+f'/{sysname:s}_ps_results.csv',index_col=0).loc[f'{sysname:s}_{name_1:s}']
            z_dil = 0.5*(np.abs(ps_results.cutoffs_dilute_left) + ps_results.cutoffs_dilute_right)
            z_den = 0.5*(np.abs(ps_results.cutoffs_dense_left) + ps_results.cutoffs_dense_right)
        if not os.path.isfile(output_path+f'/{sysname:s}_com_traj.dcd'):
            raise ValueError('Please run calc_com_traj first')
        else:
            cmtraj = md.load_dcd(output_path+f'/{sysname:s}_com_traj.dcd',top=output_path+f'/{sysname:s}_com_top.pdb')

        for chain_name, chainids in chain_indices.items():
            cm_z = cmtraj.xyz[:,chainids,2]
            mask_den = np.abs(cm_z) < z_den
            mask_dil = np.abs(cm_z) > z_dil
            rg = np.load(output_path+f'/{sysname:s}_{chain_name:s}_rg.npy')
            np.save(output_path+f'/{sysname:s}_{chain_name:s}_rg_dense.npy',rg[mask_den])
            np.save(output_path+f'/{sysname:s}_{chain_name:s}_rg_dilute.npy',rg[mask_dil])
        if name_2 == name_1:
            # if homotypic cmap, save a copy of all indices
            name_2 = name_1 + '_homotypic'
            chain_indices[name_2] = chain_indices[name_1]
        cm_z = cmtraj.xyz[:,chain_indices[name_1],2]
        # per-frame central-chain indices
        ids_central = np.argmin(np.abs(cm_z),axis=1)
        chain_indices[name_1] = np.array(
            [chain_indices[name_1][idx] for idx in ids_central]
        )

    cmap = np.zeros((N_res_1,N_res_2))
    for chain_1 in np.unique(chain_indices[name_1]):
        surrounding_chains = traj.top.select(' or '.join([f'chainid {i:d}' for i in chain_indices[name_2] if i != chain_1]))
        pair_indices = traj.top.select_pairs(f'chainid {chain_1:d}',surrounding_chains)
        if is_slab:
            mask_frames = np.where(chain_indices[name_1] == chain_1)[0]
        else:
            mask_frames = np.arange(traj.n_frames)#, True, dtype=bool)
        if len(mask_frames) > 0:
            for mf in mask_frames:
                d = md.compute_distances(traj[mf],pair_indices)[0]
                cm = (.5-.5*np.tanh((d-1.)/.3)).reshape(N_res_1,-1,N_res_2)
                cm = np.sum(cm,axis=1)
                cmap += cm
    cmap /= traj.n_frames

    # save energy and contact maps
    np.save(output_path+f'/{sysname:s}_{name_1:s}_{name_2:s}_cmap.npy',cmap)

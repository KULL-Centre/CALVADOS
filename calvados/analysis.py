import numpy as np
import numba as nb
import pandas as pd

import MDAnalysis as mda
from MDAnalysis import Merge, transformations

from MDAnalysis.analysis import distances, rms
from MDAnalysis.analysis.align import AlignTraj

import mdtraj as md

from tqdm import tqdm

from scipy.optimize import curve_fit, least_squares
from scipy.stats import sem

from calvados.build import get_ssdomains

import matplotlib.pyplot as plt

import os
import sys
from pathlib import Path
PACKAGEDIR = Path(__file__).parent.absolute()
sys.path.append(f'{str(PACKAGEDIR):s}/BLOCKING')
from main import BlockAnalysis

def center_traj(pdb,traj,start=None,stop=None,step=1):
    """ Center trajectory """
  
    u = mda.Universe(pdb,traj)
    
    with mda.Writer(f'{traj[:-4]}_c.dcd', len(u.atoms)) as W:
        for ts in u.trajectory[start:stop:step]:
            u.atoms.translate(-u.atoms.center_of_geometry() + 0.5 * u.dimensions[:3])
            W.write(u.atoms)

def subsample_traj(pdb,traj,start=None,stop=None,step=1):
    """ Subsample trajectory """

    u = mda.Universe(pdb,traj)

    with mda.Writer(f'{traj[:-4]}_sub.dcd', len(u.atoms)) as W:
        for ts in u.trajectory[start:stop:step]:
            W.write(u.atoms)

@nb.jit(nopython=True)
def calc_energy(dmap,sig,lam,rc_lj,eps_lj,qmap,
                k_yu,rc_yu=4.0,
               same_domain=False):
    """ AH and YU energ

    Input:
      * r: distance map
      * sig: sigma map
      * lam: lambda map
      * rc_lj: LJ cutoff
      * eps_lj: LJ prefactor
      * qmap: charge product map (including prefactors)
      *
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
            s0 = 2**(1./6.) * sigij

            u_ah[i,j] = ah_potential(rij,sigij,eps_lj,lamij,rc_lj)

            # YU
            q = qmap[i,j]
            if rij <= rc_yu:
                u_yu[i,j] = yukawa_potential(rij,qmap[i,j],k_yu,rc_yu=rc_yu)
            else:
                u_yu[i,j] = 0.
    return u_ah, u_yu

@nb.jit(nopython=True)
def yukawa_potential(r,q,kappa_yu,rc_yu=4.0):
    # q = epsi_yu * epsj_yu
    shift = np.exp(-kappa_yu*rc_yu)/rc_yu
    u = q * (np.exp(-kappa_yu*r)/r - shift)
    return u

@nb.jit(nopython=True)
def lj_potential(r,sig,eps):
    ulj = 4.*eps*((sig/r)**12 - (sig/r)**6)
    return ulj

@nb.jit(nopython=True)
def ah_potential(r,sig,eps,l,rc):
    if r <= 2**(1./6.)*sig:
        ah = lj_potential(r,sig,eps) - l * lj_potential(rc,sig,eps) + eps * (1 - l)
    elif r <= rc:
        ah = l * (lj_potential(r,sig,eps) - lj_potential(rc,sig,eps))
    else:
        ah = 0.
    return ah

def calc_dmap(domain0,domain1):
    """ Distance map (nm) for single configuration

    Input: Atom groups
    Output: Distance map"""
    dmap = distances.distance_array(domain0.positions, # reference
                                    domain1.positions, # configuration
                                    box=domain0.dimensions) / 10.
    return dmap

def calc_raw_dmap(pos0,pos1):
    dmap = distances.distance_array(pos0,pos1)
    return dmap

def self_distances(pos,box=None):
    """ Self distance map for matrix of positions

    If box dimensions are provided, distances are
    calculated using minimum image convention

    Input: Matrix of positions and (optional) box dimensions
    Output: Self distance map
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

def calc_wcn(comp,pos,fdomains=None,ssonly=True,r0=0.7):
    """
    pos: positions [nm]
    r0: switching parameter [nm]
    r0: switching parameter [nm] """
    N = len(pos)
    # print(f'N: {N}')
    dmap = calc_raw_dmap(pos,pos)
    # dmap = self_distances(pos)

    if ssonly:
        ssdomains = get_ssdomains(comp.name,fdomains)
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

def calc_cmap(domain0,domain1,cutoff=1.0):
     """ Contact map for single configuration

     Input: MDAnalysis Atom groups (can be the same or different)
     Output: Contact map
     """
     # Cutoff in nm
     dmap = calc_dmap(domain0,domain1)
     cmap = .5 - .5*np.tanh((dmap-cutoff)/.3)
     return(cmap)

def cmap_traj(u,domain0,domain1,cutoff=1.0,start=None,end=None,step=1):
    """ Average number of contacts along trajectory

    Input:
      * Universe
      * Atom groups
    Output:
      * Average contact map
    """
    cmap = np.zeros((len(domain0),len(domain1)))
    for ts in u.trajectory[start:end:step]:
        cmap += calc_cmap(domain0,domain1,cutoff)
    cmap /= len(u.trajectory)
    return cmap

def calc_fnc(u,uref,selstr,cutoff=1.5,kmax=1,
    bfac=[],sig_shift=0.8,width=50.):
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

def calc_rmsd(u,uref,select='all',f_out=None,step=1):
    # print('First alignment')
    aligner = AlignTraj(u, uref, select=select, in_memory=True).run(step=step) # align to crystal structure
    Rref = rms.RMSD(u,uref,select=select) # get RMSD to reference
    Rref.run(step=step)
    coords = u.trajectory.timeseries(u.atoms,step=step)
    coords_mean = coords.mean(axis=1) # get mean structure
    u_mean = Merge(u.atoms) # new universe from mean structure
    u_mean.load_new(coords_mean[:, None, :], order="afc")

    # print('Second alignment')
    aligner = AlignTraj(u, u_mean, select=select, in_memory=True).run(step=step) # align to mean structure
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

def get_masses(seq,residues,charge_termini=True):
    lseq = list(seq)
    masses = residues.loc[lseq,'MW'].values
    if charge_termini:
        masses[0] += 2.
        masses[-1] += 16.
    return masses

def calc_rg(u,ag,seq=[],residues=[],start=None,stop=None,step=None):
    if len(seq) > 0:
        masses = get_masses(seq,residues)
        # print(masses)
    else:
        masses = np.array([1. for _ in range(len(ag.atoms))])

    rogs = []
    for t, ts in enumerate(u.trajectory[start:stop:step]):
        com = ag.center(weights=masses)
        pos = (ag.positions - com) / 10.
        rog_sq = np.einsum('i,i->',masses,np.einsum('ij,ij->i',pos,pos))/np.sum(masses)
        rog = np.sqrt(rog_sq)
        rogs.append(rog)
    rogs = np.array(rogs)
    return rogs

def calc_ete(u,ag,start=None,stop=None,step=None):
    """ Mean and std of end to end distance of atom group across trajectory """
    etes = []
    # etes2 = []
    for t, ts in enumerate(u.trajectory[start:stop:step]):
        ete = np.linalg.norm(ag[0].position-ag[-1].position) / 10.
        etes.append(ete)
        # etes2.append(ete**2)
    etes = np.array(etes)
    ete_m = np.mean(etes)
    ete_sem = sem(etes)
    # ete2_m = np.mean(etes2)
    # etes = np.array(etes)
    return etes, ete_m, ete_sem#, ete2_m

def calc_ocf(u,ag,start=None,stop=None,step=None):
    """ Orientational correlation factor (OCF) as function of separation along the chain """
    ocfs = []
    for t,ts in enumerate(u.trajectory[start:stop:step]):
        x = ag.positions / 10.
        xb = x[1:] - x[:-1]
        Lxb = np.linalg.norm(xb,axis=1)
        xbred = (xb.T / Lxb).T
        dots = [[] for _ in range(len(xbred))]

        for idx0,xb0 in enumerate(xbred,start=0):
            for idx1,xb1 in enumerate(xbred[idx0:],start=idx0):
                dot = np.dot(xb0,xb1)
                ij = idx1-idx0
                dots[ij].append(dot)
        dots_avg = []
        for dot in dots:
            dots_avg.append(np.mean(dot))
        ocfs.append(dots_avg)
    ocfs = np.array(ocfs)
    ocf = np.mean(ocfs,axis=0)
    ocf_sem = sem(ocfs)
    return ocf, ocf_sem

#### SCALING EXPONENT

def scaling_exp(n,r0,v):
    rh = r0 * n**v
    return rh

def fit_scaling_exp(u,ag,r0=None,traj=True,start=None,stop=None,step=None,slic=[],ij0=5):
    """ Fit scaling exponent of single chain

    Input:
      * mda Universe
      * atom group
    Output:
      * ij seq distance
      * dij cartesian distance
      * r0
      * v (scaling exponent)
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
    dij = []
    for i in range(N):
        dij.append([])
    for i in ij:
        for j in range(i,N):
            dij[j-i].append(dmap[i,j]) # in nm

    for i in range(N):
        dij[i] = np.mean(dij[i])
    # print(dij)
    dij = np.array(dij)
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
    return ij, dij, r0, v, verr

def save_conf_prop(path,name,residues_file,output_path,start=0,is_idr=True,select='all',cutoff=1.0, kmax=3):
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
    def __init__(self,
            name,
            input_path = '.',output_path = '.',
            input_pdb = 'top.pdb', input_dcd = None,
            centered_dcd = 'traj.dcd',
            ref_chains = None, ref_name = None,
            client_chain_list = [], client_names = [],
            verbose = False
            ):
        self.name = name
        self.input_path = input_path
        self.output_path = output_path
        self.input_pdb = input_pdb
        if input_dcd is None:
            input_dcd = f'{self.name}.dcd'
        self.input_dcd = input_dcd
        self.centered_dcd = centered_dcd
        self.ref_chains = ref_chains
        self.ref_name = ref_name
        if self.ref_name is None:
            self.ref_name = 'ref'
        self.client_chain_list = client_chain_list
        self.client_names = client_names
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

    def center(self, start=None, end=None, step=1,
            center_target = 'ref'):
        """
        Center slab trajectory.
        center_target: 'ref' or 'all'. Define if particles for centering are from reference or whole system.
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
        bonds = []
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

    def calc_profiles(self, start=None, end=None, step=1,
            save_individual_profiles=True):
        """
        Calculate concentration profiles for reference chains (and possible clients).
        Keep start=None, end=None, step=1 if the centered trajectory is already cropped. """

        u = mda.Universe(f'{self.input_path}/{self.input_pdb}', f'{self.input_path}/{self.centered_dcd}', in_memory=True)

        if self.ref_chains is None:
            self.ref_chains = (0, len(u.segments)-1)
            # ag_ref = u.atoms
            # nbeads_ref = len(u.segments[0].atoms)
        # else:
        sg_ref = u.segments[self.ref_chains[0]:self.ref_chains[1]+1]
        ag_ref = sg_ref.atoms
        nbeads_ref = len(sg_ref[0].atoms)
        if self.verbose:
            print(f'Reference: name {self.ref_name}; chains {self.ref_chains[0]}-{self.ref_chains[1]}; nbeads: {nbeads_ref}')

        n_frames = len(u.trajectory[start:end:step])
        binwidth = 1 # 0.1 nm
        volume = u.dimensions[0]*u.dimensions[1]*binwidth/1e3 # volume of a slice in nm3
        conv_ref = 10/6.02214/nbeads_ref/volume*1e3 # conversion to mM

        # Reference profile
        h_ref = np.zeros((n_frames,self.n_bins))
        for t,ts in enumerate(u.trajectory[start:end:step]):
            ts = transformations.wrap(ag_ref)(ts)
            zpos = ag_ref.positions.T[2]
            h, e = np.histogram(zpos,bins=self.edges)
            h_ref[t] = h * conv_ref # mM
        if save_individual_profiles:
            np.save(f'{self.output_path}/{self.name}_{self.ref_name}_profile.npy', h_ref) # in mM

        h_ref_mean = h_ref.mean(axis=0) # mM
        self.all_profiles = [self.z_nm,h_ref_mean]

        # Client profiles
        for i, (first,last) in enumerate(self.client_chain_list):
            sg_sel = u.segments[first:last+1]
            ag_sel = sg_sel.atoms
            nbeads_sel = len(sg_sel[0].atoms)
            if self.verbose:
                print(f'Client {i}: name {self.client_names[i]}; chains {first}-{last}; nbeads: {nbeads_sel}')
            # print(nbeads_sel)
            conv_sel = 10/6.02214/nbeads_sel/volume*1e3 # conversion to mM

            h_sel = np.zeros((n_frames,self.n_bins))
            for t,ts in enumerate(u.trajectory[start:end:step]):
                # wrap for density profile calculation
                ts = transformations.wrap(ag_sel)(ts)
                zpos = ag_sel.positions.T[2]
                h, e = np.histogram(zpos,bins=self.edges)
                h_sel[t] = h * conv_sel

            if save_individual_profiles:
                np.save(f'{self.output_path}/{self.name}_{self.client_names[i]}_profile.npy', h_sel) # individual profiles
            h_sel_mean = h_sel.mean(axis=0) # mM
            self.all_profiles.append(h_sel_mean)
        self.all_profiles = np.array(self.all_profiles)

        np.save(f'{self.output_path}/{self.name}_profiles.npy', self.all_profiles) # all trajectory-averaged profiles
        if self.verbose:
            print(f'Output written to {self.output_path}/')

    def calc_concentrations(self,
            pden=2., pdil=8., dGmin=-10.,
            write_conc_arrays=True,
            input_pdb='top.pdb',
            plot_profiles=True):

        self.pden, self.pdil = pden, pdil
        self.dGmin = dGmin
        self.df_results = pd.DataFrame(dtype=object)
        self.write_conc_arrays = write_conc_arrays

        # Reference concentrations
        if self.ref_chains is None:
            u = mda.Universe(f'{self.input_path}/'+input_pdb)
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

    def save_conc_results(self, comp_name, results):
        for key, val in results.items():
            if key in ['dense_array', 'dilute_array']:
                if self.write_conc_arrays:
                    np.save(f'{self.output_path}/{comp_name}_{key}.npy', val)
            else:
                self.df_results.loc[comp_name, key] = val

    def calc_single_conc(self, h, ref=True):
        """
        Calculate dense and dilute phase concentrations.
        path: Input path.
        input_file: Concentration profile array file (e.g. A1_ref_profile.npy).
        Only specify start, end, step, if the conc. profile was from an uncropped trajectory (not default).
        Provided cutoffs_dense and cutoffs_dilute (e.g. from a reference profile) skip the cutoff calculation.
        """

        hm = np.mean(h,axis=0)

        if ref:
            self.cutoffs_dense, self.cutoffs_dilute = self.fit_profile(self.z_nm, hm, self.pden, self.pdil)

        results = {}
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

    def plot_density_profiles(self):
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

    @staticmethod
    def calc_z_Angstr(u):
        lz = u.dimensions[2]
        edges = np.arange(0,lz+1,1)
        dz = (edges[1] - edges[0]) / 2.
        z = edges[:-1] + dz
        return lz, edges, z

    @staticmethod
    def calc_z_nm_centered(u):
        lz = u.dimensions[2]
        edges = np.arange(-lz/2.,lz/2.+0.0001,1)/10
        dz = (edges[1] - edges[0]) / 2.
        z = edges[:-1] + dz
        return lz, edges, z

    @staticmethod
    def calc_zpatch(z,h):
        cutoff = 0
        ct = 0.
        ct_max = 0.
        zwindow = []
        hwindow = []
        zpatch = []
        hpatch = []
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
        zpatch = np.array(zpatch)
        hpatch = np.array(hpatch)
        return zpatch, hpatch

    @staticmethod
    def calc_dG(c_dil,e_dil,c_den,e_den,ndraws=10000,dGmin=-10):
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
            spread_dGs = []
            for idraw, (dil,den) in enumerate(zip(spread_dil,spread_den)):
                if dil > 0 and den > 0:
                    spread_dGs.append(np.log(dil/den))
            dG_error = np.std(spread_dGs)
        if dG < dGmin:
            dG = dGmin
            dG_error = 0.
            print(f'dG extremely small, setting dG to {dGmin:.1f}')
        return dG, dG_error

    @staticmethod
    def fit_profile(z, hm, pden, pdil):
        profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d) # hyperbolic function, parameters correspond to csat etc.
        residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )
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
    def calc_block_errors(denarray, dilarray):
        """ Block error analysis """

        block_den = BlockAnalysis(denarray)
        block_dil = BlockAnalysis(dilarray)

        block_den.SEM()
        block_dil.SEM()

        eden = block_den.sem
        edil = block_dil.sem

        return eden, edil

def calc_com_traj(path,sysname,output_path,residues_file,chainid_dict={},start=None,end=None,step=1,input_pdb='top.pdb'):
    """
    Calculate trajectory of chain COMs and per-frame Rg's for each chain.

    Parameters:
    -----------
    chainid_dict : dict
        Examples:
            {'name_1': 0, 'name_2': 1}
            {'name_1': (0, 99), 'name_2': (100, 199)}
        - Keys are component names.
        - Values are integers or tuples representing the first and last chain IDs.

        The dictionary can contains as many entries as the number of components in the system.

        If the dictionary is not provided as an argument, the function assumes a
        single-component system named `sysname` and calculates a COM trajectory and per-frame Rg's
        for all the chains in the topology.
    """
    if not os.path.isfile(f'{path:s}/traj.dcd'):
        u = mda.Universe(f'{path:s}/{input_pdb:s}',f'{path:s}/{sysname:s}.dcd',in_memory=True)
        ag = u.select_atoms('all')
        n_atoms = ag.n_atoms
        # create list of bonds
        bonds = []
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

    chain_prop = {}
    n_chains = 0
    for chain_name, chainids in chainid_dict.items():
        chain_prop[chain_name] = {}
        if type(chainids) is int:
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
        for chainid in chain_prop[chain_name]['ids']:
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

def calc_contact_map(path,sysname,output_path,chainid_dict={},is_slab=False,input_pdb='top.pdb'):
    """
    Calculate the contact map between two sets of chain IDs specified in the given dictionary.

    Parameters:
    -----------
    chainid_dict : dict
        Examples:
            {'name_1': 0, 'name_2': 1}
            {'name_1': (0, 99), 'name_2': (100, 199)}
        - Keys are component names.
        - Values are integers or tuples representing the first and last chain IDs.

        If the dictionary contains only one chain entry, the function calculates a
        homotypic contact map.
        If the dictionary is not provided as an argument, the function assumes a
        single-component system named `sysname` and calculates a homotypic contact using
        all the chains in the topology.

    is_slab : bool, optional (default=False)
        If True, the function calculates a contact map between chains in the midplane
        of the slab and all surrounding chains.
        In this case, the first item in `chainid_dict` should be the component
        used to center the slab in `SlabAnalysis`.
    """
    traj = md.load_dcd(f'{path:s}/traj.dcd',top=f'{path:s}/'+input_pdb)
    traj.xyz -= traj.unitcell_lengths[0,:]/2

    if len(chainid_dict) > 0:
        name_1 = next(iter(chainid_dict))
        if type(chainid_dict[name_1]) is int:
            chainid_dict[name_1] = (chainid_dict[name_1], chainid_dict[name_1])
        chainid_dict[name_1] = np.arange(chainid_dict[name_1][0], chainid_dict[name_1][1]+1)
        if len(chainid_dict) > 1:
            name_2 = next(iter(list(chainid_dict.keys())[1:]))
            if type(chainid_dict[name_2]) is int:
                chainid_dict[name_2] = (chainid_dict[name_2], chainid_dict[name_2])
            chainid_dict[name_2] = np.arange(chainid_dict[name_2][0], chainid_dict[name_2][1]+1)
        else:
            # if homotypic cmap
            name_2 = name_1
    else:
        name_1 = sysname
        name_2 = name_1
        chainid_dict[name_1] = np.arange(traj.top.n_chains)

    N_res_1 = traj.top.chain(chainid_dict[name_1][0]).n_residues
    N_res_2 = traj.top.chain(chainid_dict[name_2][0]).n_residues

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

        for chain_name, chainids in chainid_dict.items():
            cm_z = cmtraj.xyz[:,chainids,2]
            mask_den = np.abs(cm_z) < z_den
            mask_dil = np.abs(cm_z) > z_dil
            rg = np.load(output_path+f'/{sysname:s}_{chain_name:s}_rg.npy')
            np.save(output_path+f'/{sysname:s}_{chain_name:s}_rg_dense.npy',rg[mask_den])
            np.save(output_path+f'/{sysname:s}_{chain_name:s}_rg_dilute.npy',rg[mask_dil])
        if name_2 == name_1:
            # if homotypic cmap, save a copy of all indices
            name_2 = name_1 + '_homotypic'
            chainid_dict[name_2] = chainid_dict[name_1]
        cm_z = cmtraj.xyz[:,chainid_dict[name_1],2]
        # per-frame central-chain indices
        chainid_dict[name_1] = np.argmin(np.abs(cm_z),axis=1)

    cmap = np.zeros((N_res_1,N_res_2))
    for chain_1 in np.unique(chainid_dict[name_1]):
        surrounding_chains = traj.top.select(' or '.join([f'chainid {i:d}' for i in chainid_dict[name_2] if i != chain_1]))
        pair_indices = traj.top.select_pairs(f'chainid {chain_1:d}',surrounding_chains)
        if is_slab:
            mask_frames = chainid_dict[name_1] == chain_1
        else:
            mask_frames = np.full(traj.n_frames, True, dtype=bool)
        if np.any(mask_frames):
            d = md.compute_distances(traj[mask_frames],pair_indices)
            cmap += (.5-.5*np.tanh((d-1.)/.3)).reshape(mask_frames.sum(),
                        N_res_1,-1,N_res_2).sum(axis=0).sum(axis=1)
    cmap /= traj.n_frames
    # save energy and contact maps
    np.save(output_path+f'/{sysname:s}_{name_1:s}_{name_2:s}_cmap.npy',cmap)

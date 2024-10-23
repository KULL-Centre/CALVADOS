import numpy as np
import numba as nb
import pandas as pd

import MDAnalysis
from MDAnalysis import Merge
from MDAnalysis import transformations
from MDAnalysis.analysis import distances, rms
from MDAnalysis.analysis.align import AlignTraj

import mdtraj as md

from scipy.optimize import curve_fit
from scipy.stats import sem

from calvados.build import get_ssdomains

import sys
from pathlib import Path
PACKAGEDIR = Path(__file__).parent.absolute()
sys.path.append(f'{str(PACKAGEDIR):s}/BLOCKING')
from main import BlockAnalysis

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

def self_distances(pos):
    """ Self distance map for matrix of positions

    Input: Matrix of positions
    Output: Self distance map
    """
    N = len(pos)
    dmap = np.zeros((N,N))
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
                if fdomains != None:
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

def calc_cmap(domain0,domain1,cutoff=1.5):
    """ Contact map for single configuration

    Input: Atom groups
    Output: Contact map
    """
    # Cutoff in nm
    dmap = calc_dmap(domain0,domain1)
    cmap = np.where(dmap<cutoff,1,0)
    return(cmap)

def cmap_traj(u,domain0,domain1,cutoff=1.5):
    """ Average number of contacts along trajectory

    Input:
      * Universe
      * Atom groups
    Output:
      * Average contact map
    """
    cmap = np.zeros((len(domain0),len(domain1)))
    for ts in u.trajectory:
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

    if f_out != None:
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
    if r0 == None:
        (r0, v), pcov = curve_fit(scaling_exp,ij[ij0:],dij[ij0:])
        perr = np.sqrt(np.diag(pcov))
        verr = perr[1]
        # print(pcov)
    else:
        v, pcov = curve_fit(lambda x, v: scaling_exp(x,r0,v), ij[ij0:], dij[ij0:])
        v = v[0]
        perr = np.sqrt(np.diag(pcov))
        verr = perr[0]
    return ij, dij, r0, v, verr

def save_rg(path,name,residues_file,output_prefix,nskip,select='all'):
    residues = pd.read_csv(residues_file).set_index('three')
    u = MDAnalysis.Universe(f'{path:s}/top.pdb',f'{path:s}/{name:s}.dcd',in_memory=True)
    ag = u.select_atoms(select)
    rgs = calc_rg(u,ag,ag.resnames.tolist(),residues,start=nskip)
    block_rg = BlockAnalysis(rgs)
    block_rg.SEM()
    df_analysis = pd.DataFrame(index=['Rg'],columns=['value','error'])
    df_analysis.loc['Rg','value'] = np.mean(rgs)
    df_analysis.loc['Rg','error'] = block_rg.sem
    df_analysis.to_csv(output_prefix+'/Rg.csv')

def save_ree(path,name,output_prefix,nskip,select='all'):
    u = MDAnalysis.Universe(f'{path:s}/top.pdb',f'{path:s}/{name:s}.dcd',in_memory=True)
    ag = u.select_atoms(select)
    rees, _, _ = calc_ete(u,ag,start=nskip)
    block_ree = BlockAnalysis(rees)
    block_ree.SEM()
    df_analysis = pd.DataFrame(index=['Ree'],columns=['value','error'])
    df_analysis.loc['Ree','value'] = np.mean(rees)
    df_analysis.loc['Ree','error'] = block_ree.sem
    df_analysis.to_csv(output_prefix+'/Ree.csv')

def save_nu(path,name,output_prefix,nskip,select='all'):
    u = MDAnalysis.Universe(f'{path:s}/top.pdb',f'{path:s}/{name:s}.dcd',in_memory=True)
    ag = u.select_atoms(select)
    _, _, _, nu, nu_err = fit_scaling_exp(u,ag,start=nskip)
    df_analysis = pd.DataFrame(index=['nu'],columns=['value','error'])
    df_analysis.loc['nu','value'] = nu
    df_analysis.loc['nu','error'] = nu_err
    df_analysis.to_csv(output_prefix+'/nu.csv')

def convert_h(hs,Lseq,Lxy,mw):
    """
    Convert slab histograms to mM and mg/ml
    Requires that box x==y
    Lxy in nm !
    """
    conv = 1e5/6.022/Lseq/Lxy/Lxy
    hconv_mM = hs * conv # convert to protein concentration [mmol/L]
    hconv_mgml = hconv_mM * 1e-3 * mw # convert to protein concentration [mg/mL]
    return hconv_mM, hconv_mgml

def calc_dG(c_dil,e_dil,c_den,e_den,ndraws=10000):
    dG = np.log(c_dil/c_den)
    spread_dil = np.random.normal(c_dil,e_dil,size=ndraws)
    spread_den = np.random.normal(c_den,e_den,size=ndraws)
    # spread_dGs = np.zeros(ndraws)
    spread_dGs = []
    for idraw, (dil,den) in enumerate(zip(spread_dil,spread_den)):
        if dil > 0 and den > 0:
            spread_dGs.append(np.log(dil/den))

    dG_error = np.std(spread_dGs)
    return dG, dG_error

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

def center_slab(path,name,ref_atoms='all',start=None,end=None,step=1,input_pdb='top.pdb'):
    u = MDAnalysis.Universe(f'{path:s}/{input_pdb:s}',f'{path:s}/{name:s}.dcd',in_memory=True)
    n_frames = len(u.trajectory[start:end:step])
    ag_ref = u.select_atoms(ref_atoms)
    ag = u.select_atoms('all')
    n_atoms = ag.n_atoms
    # create list of bonds
    bonds = []
    for segment in u.segments:
        for i in segment.atoms.indices[:-1]:
            bonds.extend([(i, i+1)])
    u.add_TopologyAttr('bonds', bonds)
    L = u.dimensions[0]/10
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    hs = np.zeros((n_frames,n_bins))
    with MDAnalysis.Writer(f'{path:s}/traj.dcd',n_atoms) as W:
        for t,ts in enumerate(u.trajectory[start:end:step]):
            # shift max density to center
            zpos = ag_ref.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            zmax = z[np.argmax(h)]
            #ag.translate(np.array([0,0,-zmax+0.5*lz]))
            ag.translate(np.array([0,0,-zmax+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            #ts = transformations.wrap(ag_ref)(ts)
            zpos = ag_ref.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zpatch, hpatch = calc_zpatch(z,h)
            zmid = np.average(zpatch,weights=hpatch)
            ag.translate(np.array([0,0,-zmid+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag_ref.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            hs[t] = h
            # make chains whole
            ts = transformations.unwrap(ag)(ts)
            W.write(ag)
    return hs, z/10.

def calc_slab_profiles(path,name,ref_atoms,sel_atoms_list,output_folder,start=None,end=None,step=1,input_pdb='top.pdb'):
    """
    path: path where trajectory and pdb are saved
    ref_atoms: reference atoms to shift to the middle of the box
    sel_atoms_list: list of extra atoms for which we calculate the density profile
    output_folder: folder where output files are saved
    nskip: number of frames to skip to reach equilibrium
    """
    # center trajectory based on reference beads
    h_ref, z = center_slab(path,name,ref_atoms,start,end,step,input_pdb)
    # load centered trajectory
    u = MDAnalysis.Universe(f'{path:s}/'+input_pdb,f'{path:s}/traj.dcd',in_memory=True)
    n_frames = len(u.trajectory[start:end:step])
    h_ref = h_ref[start:end:step]
    binwidth = 1 # 0.1 nm
    # volume of a slice in nm3
    volume = u.dimensions[0]*u.dimensions[1]*binwidth/1e3
    # density profile for ref_atoms
    edges = np.arange(0,z.size+binwidth,binwidth)
    np.save(output_folder+f'/{name:s}_ref_profile.npy',np.c_[h_ref/volume.mean()])
    h_ref_mean = h_ref.mean(axis=0)/volume.mean() # number of beads per nm3
    n_bins = edges.size - 1
    all_profiles = np.c_[z,h_ref_mean]
    # density profile: selected atoms
    for i,sel_atoms in enumerate(sel_atoms_list):
        ag_sel = u.select_atoms(sel_atoms)
        h_sel = np.zeros((n_frames,n_bins))
        for t,ts in enumerate(u.trajectory[start:end:step]):
            zpos = ag_sel.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            h_sel[t] = h
        np.save(output_folder+f'/{name:s}_sel_profile_{i:d}.npy',np.c_[h_sel/volume.mean()])
        h_sel_mean = h_sel.mean(axis=0)/volume.mean() # number of beads per nm3
        all_profiles = np.c_[all_profiles,h_sel_mean]

    np.save(output_folder+f'/{name:s}_profiles.npy',all_profiles)

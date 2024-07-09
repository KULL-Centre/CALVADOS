import pandas as pd
import numpy as np
import mdtraj as md
import itertools
import os
import MDAnalysis
from MDAnalysis import transformations

def init_proteins():
    proteins = pd.DataFrame(columns=['eps_factor','pH','ionic','fasta'])
    fasta_A1 = """GSMASASSSQRGRSGSGNFGGGRGGGFGGNDNFGRGGNFSGRGGFGGSRGGGGYGGSGDGYNGFGNDGSNFGGGGSYNDFGNYNNQ
SSNFGPMKGGNFGGRSSGGSGGGGQYFAKPRNQGGYGGSSSSSSYGSGRRF""".replace('\n', '')

    proteins.loc['A1'] = dict(eps_factor=0.2,pH=7.0,fasta=fasta_A1,ionic=0.15)
    return proteins

def genParamsLJ(df,name,prot):
    fasta = list(prot.fasta)
    r = df.copy()
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['X','MW'] += 2
    r.loc['Z','MW'] += 16
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    types = list(np.unique(fasta))
    MWs = [r.loc[a,'MW'] for a in types]
    lj_eps = prot.eps_factor*4.184
    return lj_eps, fasta, types, MWs

def genParamsDH(df,name,prot,temp):
    kT = 8.3145*temp*1e-3
    fasta = list(prot.fasta)
    r = df.copy()
    # Set the charge on HIS based on the pH of the protein solution
    r.loc['H','q'] = 1. / ( 1 + 10**(prot.pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['Z'] = r.loc[fasta[-1]]
    fasta[0] = 'X'
    fasta[-1] = 'Z'
    r.loc['X','q'] = r.loc[prot.fasta[0],'q'] + 1.
    r.loc['Z','q'] = r.loc[prot.fasta[-1],'q'] - 1.
    # Calculate the prefactor for the Yukawa potential
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/kT
    yukawa_eps = [r.loc[a].q*np.sqrt(lB*kT) for a in fasta]
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*prot.ionic*6.022/10)
    return yukawa_eps, yukawa_kappa

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

def center_slab(name,start=None,end=None,step=1,input_pdb='top.pdb'):
    path = f'{name:s}'
    u = MDAnalysis.Universe(f'{path:s}/{input_pdb:s}',f'{path:s}/{name:s}.dcd',in_memory=True)
    n_frames = len(u.trajectory[start:end:step])
    ag = u.atoms
    n_atoms = ag.n_atoms
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
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            zmax = z[np.argmax(h)]
            ag.translate(np.array([0,0,-zmax+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zpatch, hpatch = calc_zpatch(z,h)
            zmid = np.average(zpatch,weights=hpatch)
            ag.translate(np.array([0,0,-zmid+0.5*lz]))
            ts = transformations.wrap(ag)(ts)
            zpos = ag.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            hs[t] = h
            W.write(ag)
    np.save(f'{name:s}_conc.npy',hs,allow_pickle=False)
    return hs, z

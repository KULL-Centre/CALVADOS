
import numpy as np
import MDAnalysis as mda
from MDAnalysis import transformations

from tqdm import tqdm

def center_traj(pdb,traj):
    """ center trajectory """
  
    u = mda.Universe(pdb,traj)
    
    with mda.Writer(f'{traj[:-4]}_c.dcd', len(u.atoms)) as W:
        for ts in u.trajectory:
            u.atoms.translate(-u.atoms.center_of_geometry() + 0.5 * u.dimensions[:3])
            W.write(u.atoms)

def subsample_traj(pdb,traj,start=None,stop=None,step=1):
    """ subsample trajectory """
    u = mda.Universe(pdb,traj)

    with mda.Writer(f'{traj[:-4]}_sub.dcd', len(u.atoms)) as W:
        for ts in u.trajectory[start:stop:step]:
            W.write(u.atoms)

def calc_zpatch(z,h,cutoff=0):
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
    if ct > ct_max: # edge case (slab at side of box)
        zpatch = zwindow
        hpatch = hwindow
    zpatch = np.array(zpatch)
    hpatch = np.array(hpatch)
    return zpatch, hpatch

def center_slab(path,name,temp,start=None,end=None,step=1,input_pdb='top.pdb',
        selstr_ref='all',selstr_out='all',fout=None,write_traj=True,
        subsampled=False,calc_coms=False):
    if subsampled:
        substring = "_sub"
    else:
        substring = ""
    u = mda.Universe(f'{path}/{input_pdb}',path+f'/{name}{substring}.dcd',in_memory=True)

    n_frames = len(u.trajectory[start:end:step])

    ag = u.atoms # all atoms (also used for dcd output)
    n_atoms = ag.n_atoms

    if calc_coms:
        segs = [seg.atoms for seg in u.segments]

    ag_ref = u.select_atoms(selstr_ref) # reference group for centering

    # print(u.dimensions)
    lz = u.dimensions[2]
    edges = np.arange(0,lz+1,1)
    dz = (edges[1] - edges[0]) / 2.
    z = edges[:-1] + dz
    n_bins = len(z)
    nframes = len(u.trajectory[start:end:step])

    # select group for npy output
    if isinstance(selstr_out,str):
        ag_out = u.select_atoms(selstr_out) 
        hs = np.zeros((n_frames,n_bins))
    elif isinstance(selstr_out,list):
        ag_out = [u.select_atoms(s) for s in selstr_out]
        hs = [np.zeros((n_frames,n_bins)) for _ in selstr_out]

    if calc_coms:
        comstraj = []

    with mda.Writer(path+'/traj.dcd',n_atoms) as W:
        for t,ts in tqdm(enumerate(u.trajectory[start:end:step]),
        total=nframes, mininterval=1):
            # calc coms
            if calc_coms:
                coms = np.array([seg.center_of_mass()[2] for seg in segs])
            # shift max density to center of ag_ref
            zpos = ag_ref.positions.T[2]
            h, e = np.histogram(zpos,bins=edges)
            zmax = z[np.argmax(h)]
            ag.translate(np.array([0,0,-zmax+0.5*lz]))
            if calc_coms:
                coms = coms - zmax + 0.5*lz
            ts = transformations.wrap(ag)(ts) # wrap
            # adjust center through weighted average of ag_ref density
            zpos = ag_ref.positions.T[2]
            h, e = np.histogram(zpos, bins=edges)
            zpatch, hpatch = calc_zpatch(z,h)
            zmid = np.average(zpatch,weights=hpatch) # center of mass of slab
            ag.translate(np.array([0,0,-zmid+0.5*lz]))
            if calc_coms:
                coms = coms - zmid + 0.5*lz
                coms -= (coms // lz) * lz
                coms -= 0.5*lz
                coms /= 10.
                comstraj.append(coms)
            ts = transformations.wrap(ag)(ts) # wrap
            # write dcd of all atoms
            if write_traj:
                W.write(ag)
            # density histogram for selected atoms
            if isinstance(selstr_out,str):
                zpos = ag_out.positions.T[2]
                h, e = np.histogram(zpos,bins=edges)
                hs[t] = h
            elif isinstance(selstr_out,list): # several output strings
                for idx,ago in enumerate(ag_out):
                    if len(ago.positions) != 0:
                        zpos = ago.positions.T[2]
                        h, e = np.histogram(zpos,bins=edges)
                        hs[idx][t] = h
    if fout == None:
        fout = f'{name:s}_{temp:d}'
    if isinstance(selstr_out,str):
        np.save(path+f'/{fout}.npy', hs, allow_pickle=False)
    elif isinstance(selstr_out,list):
        for idx,h in enumerate(hs):
            if np.sum(h) != 0:
                np.save(path+f'/{fout[idx]}.npy', h, allow_pickle=False)
    if calc_coms:
        comstraj = np.array(comstraj)
        np.save(path+'/coms.npy',comstraj)
    return hs, z
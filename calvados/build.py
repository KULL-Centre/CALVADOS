from scipy import constants
import numpy as np
import pandas as pd
from openmm import unit

from MDAnalysis import Universe
from MDAnalysis.analysis import distances

from json import load
from yaml import safe_load

from .sequence import calc_mw

from warnings import catch_warnings, simplefilter


################ SYSTEM BUILDING FUNCTIONS ################

def build_box(Lx,Ly,Lz):
    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = Lx * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = Ly * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = Lz * unit.nanometers
    return a, b, c

def calc_box(N):
    if N > 350:
        box = [25., 25., 300.] # nm
    else:
        box = [20., 20., 200.] # nm
    return box

def calc_nprot_slab(N,box,pbeads=90):
    ''' pbeads: beads per nm^2 in xy directions '''
    beads = pbeads * box[0] * box[1]
    nprot = int(beads / N)
    return nprot

def check_walls(x,box):
    """ x must be between 0 and [Lx,Ly,Lz] """
    if np.min(x) < 0: # clash with left box wall
        return True
    xbox = box
    d = xbox - x
    if np.min(d) < 0:
        return True # clash with right box wall
    return False # molecule in box

def check_clash(x,pos,box,cutoff=0.7):
    """ Check for clashes with other particles.
    Returns true if clash, false if no clash. """
    boxfull = np.append(box,[90,90,90])
    xothers = np.array(pos)
    if len(xothers) == 0:
        return False # no other particles
    d = distances.distance_array(x,xothers,boxfull)
    if np.amin(d) < cutoff:
        return True # clash with other particles
    else:
        return False # no clash

def draw_vec(l,ndim=3):
    """
    draw unbiased random vector with length l
    """
    while True:
        vec = np.random.random(size=ndim) - 0.5
        if np.linalg.norm(vec) < 0.5:
            break
    vecnorm = vec / np.linalg.norm(vec)
    vecL = vecnorm * l
    return vecL

def draw_starting_vec(box):
    """
    draw random position within the simulation box
    """
    vec = np.random.random(size=3) * box
    return vec

def build_linear(z_bondlengths, n_per_res=1, ys=None):
    """
    linear chain growing in z direction with possible extra beads at y offset n_per_res*ys
    centered at [0,0,0]
    """
    N = len(z_bondlengths)
    zs = np.zeros(N)
    zs[1:] = [(z_bondlengths[idx] + z_bondlengths[idx+1]) / 2 for idx in range(N-1)]
    zs = np.cumsum(zs)
    coords = np.zeros((N*n_per_res,3))
    if ys == None:
        ys = np.zeros(N)
    for idx, (z, y) in enumerate(zip(zs,ys)):
        for jdx in range(n_per_res):
            coords[idx*n_per_res + jdx] = [0,jdx*y,z]
    coords[:,2] -= coords[:,2].mean()
    return coords

def p2c(r, phi):
    """
    polar to cartesian
    """
    return (r * np.cos(phi), r * np.sin(phi))

def build_spiral(bondlengths, delta=0, arc=.38, separation=.7, n_per_res=1):
    """
    create points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive
    turnings
    """
    r = arc
    b = separation / (2 * np.pi)
    phi = float(r) / b
    coords = []
    for i,z in enumerate(bondlengths):
        for j in range(n_per_res): # number of beads per residue (placed along z with bondlength)
            coords.append(list(p2c(r, phi))+[j*z]) # j*z = 0 for n_per_res=1
        phi += float(arc) / r
        r = b * phi
    return np.array(coords)+delta

def build_compact(nbeads, d=0.38, verbose=False):
    N = int(np.ceil(np.cbrt(nbeads)) - 1)
    if verbose:
        print(f'Building {N+1} * {N+1} * {N+1} grid.')
    xs = []
    i, j, k = 0, 0, 0
    di, dj, dk = 1, 1, 1 # direction
    cti, ctj, ctk = 0, 0, 0

    for idx in range(nbeads):
        xs.append([i,j,k])
        if ctk == N:
            if ctj == N:
                i += di
                cti += 1
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
    xs = (np.array(xs) - 0.5*N) * d
    return xs

def random_placement(box,xs_others,xinit,ntries=10000):
    ntry = 0
    while True: # random placement
        ntry += 1
        if ntry > ntries:
            raise ValueError(f"Tried {ntries}x to add molecule. Giving up.")
        x0 = draw_starting_vec(box) # random point in box
        xs = x0 + xinit
        walls = check_walls(xs,box) # check if outside box
        if walls:
            continue
        clash = check_clash(xs,xs_others,box) # check if clashes with existing pos
        if clash:
            continue
        else:
            break
    return xs

def build_xybilayer(x0,box,xs_others,xinit,upward=True):
    inserted = True
    xs = x0 + xinit
    xs[:,2] -= xs[1,2]
    xs[:,2] += box[2]/2 - 1.5
    if upward:
        xs = xs[::-1,:]
        xs[:,2] += 3
        walls = check_walls(xs,box) # check if outside box
        if walls:
            inserted = False
        clash = check_clash(xs,xs_others,box,cutoff=0.5) # check if clashes with existing pos
        if clash:
            inserted = False
    else:
        walls = check_walls(xs,box) # check if outside box
        if walls:
            inserted = False
        clash = check_clash(xs,xs_others,box,cutoff=0.5) # check if clashes with existing pos
        if clash:
            inserted = False
    return xs, inserted

def build_xygrid(N,box,z=0.):
    """ Grid for xy slabs """
    if np.sqrt(N) % 1 > 0:
        b = 2
    else:
        b = 1
    nx = int(np.sqrt(N)) + b # nx spots in x dim
    ny = int(np.sqrt(N)) + b # ny spots in x dim

    dx = box[0] / nx
    dy = box[1] / ny

    xy = []
    x, y = 0., 0.
    ct = 0
    for n in range(N):
        ct += 1
        xy.append([x,y,z])
        if ct == ny:
            y = 0
            x += dx
            ct = 0
        else:
            y += dy
    xy = np.array(xy)
    return xy

def build_xyzgrid(N,box):
    """ 3D grid """

    r = box / np.sum(box)
    a = np.cbrt(N / np.product(r))
    n = a * r
    nxyz = np.floor(n)
    while np.product(nxyz) < N:
        ndeviation = n / nxyz
        devmax = np.argmax(ndeviation)
        nxyz[devmax] += 1
    while np.product(nxyz) > N:
        nmax = np.argmax(nxyz)
        nxyz[nmax] -= 1
        if np.product(nxyz) < N:
            nxyz[nmax] += 1
            break

    xyz = []
    x, y, z = 0., 0., 0.

    ctx, cty, ctz = 0, 0, 0

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
            xshift = dx/2
            yshift = dy/2

        if xyplane < 0:
            zshift = dz/2
        else:
            zshift = 0

        xyz.append([x+xshift,y+yshift,z+zshift])

        ctx += 1
        x += dx

        if (ctx % 2 == cty % 2):
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
                x = 0.
                y = 0.

                ctz += 1
                z += dz

                zplane = -zplane

    xyz = np.asarray(xyz)
    return xyz

# FOLDED
def geometry_from_pdb(pdb,use_com=False):
    """ positions in nm"""
    with catch_warnings():
        simplefilter("ignore")
        u = Universe(pdb)
    ag = u.atoms
    ag.translate(-ag.center_of_mass())
    if use_com:
        coms = []
        for res in u.residues:
            com = res.atoms.center_of_mass()
            coms.append(com)
        pos = np.array(coms) / 10.
    else:
        cas = u.select_atoms('name CA')
        pos = cas.positions / 10.
    return pos, u.dimensions

def bfac_from_pdb(pdb,confidence=70.):
    """ get pLDDT encoded in pdb b-factor column """
    with catch_warnings():
        simplefilter("ignore")
        u = Universe(pdb)
    bfac = np.zeros((len(u.residues)))
    for idx, res in enumerate(u.residues):
        bfac[idx] = np.mean(res.atoms.tempfactors) # average b-factor for residue
    bfac = np.where(bfac>confidence,bfac,0.) / 100. # high confidence filter
    return bfac

def load_pae_inv(input_pae,cutoff=0.1,colabfold=0,symmetrize=True):
    """ pae as json file (AF2 format) """
    pae = load_pae(input_pae,colabfold=colabfold,symmetrize=True)
    pae = np.where(pae < 1., 1, pae) # avoid division by zero (for i = j), min to 1
    pae_inv = 1/pae # inverse pae
    pae_inv = np.where(pae_inv > cutoff, pae_inv, 0)
    return pae_inv

def load_pae(input_pae,colabfold=0,symmetrize=True):
    """ pae as json file (AF2 format) """
    with open(input_pae) as f:
        pae = load(f)
        if colabfold == 0:
            pae = np.array(pae[0]['predicted_aligned_error'])
        elif colabfold == 1:
            pae = np.array(pae['predicted_aligned_error'])
        elif colabfold == 2:
            pae = np.array(pae['pae'])
    if symmetrize:
        pae = 0.5 * (pae + pae.T)
    return pae

def get_ssdomains(name,fdomains,dpam=False):
    if dpam:
        domains = []
        df_dpam = pd.read_csv(fdomains,delimiter='\t').set_index('uniprot')
        for key, val in df_dpam.iterrows():
            if key == name:
                rng = val['range'].split('-')
                domains.append([int(rng[0]),int(rng[1])])
    else:
        with open(f'{fdomains}','r') as f:
            stream = f.read()
            domainbib = safe_load(stream)
        domains = domainbib[name]

    print(f'Using domains {domains}')

    ssdomains = []
    for didx, domain in enumerate(domains):
        xs = [] # restraint residues of domain
        if isinstance(domain[0],list):
            for subdom in domain:
                for x in range(subdom[0]-1,subdom[1]):
                    xs.append(x)
        else:
            for x in range(domain[0]-1,domain[1]):
                xs.append(x)
        ssdomains.append(xs) # use 1-based, inclusive
    return ssdomains

def check_ssdomain(ssdomains,i,j,req_both=True):
    """
    Check if one (req_both == False) or both (req_both == True) of
    the residues are in a structured domain.
    0-based
    """

    ss = False
    for ssdom in ssdomains:
        if req_both:
            if (i in ssdom) and (j in ssdom):
                ss = True
        else:
            if (i in ssdom) or (j in ssdom):
                ss = True
    return ss

def conc_to_n(cinp,p,V,mw):
    """ g/L to n """
    n = p * 1e-24 * 1/mw * cinp * V * constants.N_A
    return n

def n_to_conc(n,V,mw):
    """ n to g/L """
    c = n * mw * 1/constants.N_A * 1e24 * 1/V
    return c

def calc_pair_n_in_box(cinp,pB,box,seqA,seqB):
    """ protein numbers in simulation box of two species """
    pA = 1.-pB
    V = box[0]*box[1]*box[2] # nm^3
    mwA = calc_mw(seqA) # Da = g/mol
    mwB = calc_mw(seqB) # Da = g/mol

    nA = round(conc_to_n(cinp,pA,V,mwA))
    nB = round(conc_to_n(cinp,pB,V,mwB))

    mA_rounded = nA*mwA
    mB_rounded = nB*mwB

    pB_rounded = mB_rounded / (mA_rounded + mB_rounded)
    c_rounded = n_to_conc(nA,V,mwA) + n_to_conc(nB,V,mwB)
    return nA, nB, pB_rounded, c_rounded

def calc_mixture_n_in_box(cinp,ps,box,seqs):
    """
    Protein numbers in simulation box of several species
    ps: list of mass proportions
    seqs: list of sequences
    """
    V = box[0]*box[1]*box[2] # nm^3

    ns = []
    mws = []

    for p, seq in zip(ps, seqs):
        mw = calc_mw(seq) # Da = g/mol
        mws.append(mw)
        n = round(conc_to_n(cinp,p,V,mw)) # number of proteins per type
        ns.append(n)

    cs_rounded = np.array([n_to_conc(n,V,mw) for n, mw in zip(ns,mws)]) # rounded g/L per type
    ctotal = np.sum(cs_rounded) # total g/L mass conc
    ps_rounded = cs_rounded / ctotal # mass fractions

    return ns, ps_rounded, ctotal

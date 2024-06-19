import hoomd
import hoomd.md
from hoomd import azplugins
import time
import itertools
import pandas as pd
import numpy as np
import mdtraj as md
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--seq_name',nargs='?',required=True)
parser.add_argument('--path',nargs='?',required=True)
args = parser.parse_args()

print(hoomd.__file__)

def xy_spiral_array(n, delta=0, arc=.38, separation=.7):
    """
    create points on an Archimedes' spiral
    with `arc` giving the length of arc between two points
    and `separation` giving the distance between consecutive
    turnings
    """
    def p2c(r, phi):
        """
        polar to cartesian
        """
        return (r * np.cos(phi), r * np.sin(phi))
    r = arc
    b = separation / (2 * np.pi)
    phi = float(r) / b
    coords = []
    for i in range(n):
        coords.append(list(p2c(r, phi))+[0])
        phi += float(arc) / r
        r = b * phi
    return np.array(coords)+delta

def gen_init_top(residues,fasta,path,L):
    N_res = len(fasta)
    top = md.Topology()
    chain = top.add_chain()
    for resname in fasta:
        residue = top.add_residue(residues.loc[resname,'three'], chain)
        top.add_atom('CA', element=md.element.carbon, residue=residue)
    for i in range(N_res-1):
        top.add_bond(top.atom(i),top.atom(i+1))
    pos = xy_spiral_array(N_res)
    t = md.Trajectory(np.array(pos).reshape(N_res,3), top, 0, [L,L,L], [90,90,90])
    t.save_pdb(path+'/init_top.pdb')

def gen_params(r,seq,temp,ionic):
    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*ionic*6.022/10)
    fasta = list(seq.fasta)
    r.loc['H','q'] = 1. / ( 1 + 10**(seq.pH-6) )
    r.loc['X'] = r.loc[fasta[0]]
    r.loc['X','q'] = r.loc[fasta[0],'q'] + 1.
    r.loc['X','MW'] = r.loc[fasta[0],'MW'] + 2.
    fasta[0] = 'X'
    r.loc['Z'] = r.loc[fasta[-1]]
    r.loc['Z','q'] = r.loc[fasta[-1],'q'] - 1.
    r.loc['Z','MW'] = r.loc[fasta[-1],'MW'] + 16.
    fasta[-1] = 'Z'
    # Calculate the prefactor for the Yukawa potential
    qq = pd.DataFrame(r.q.values*r.q.values.reshape(-1,1),index=r.q.index,columns=r.q.index)
    yukawa_eps = qq*lB*RT
    types = list(np.unique(fasta))
    pairs = np.array(list(itertools.combinations_with_replacement(types,2)))
    return yukawa_kappa, yukawa_eps, types, pairs, fasta, r

def gen_xtc(path):
    traj = md.load_dcd(path+'/traj.dcd', top=path+'/init_top.pdb')
    traj.xyz *= 10
    traj.unitcell_lengths *= 10
    traj.center_coordinates()
    traj[-1].save_pdb(path+'/top.pdb')
    #skip first 10 frames
    traj[10:].save_xtc(path+'/traj.xtc')

def simulate(residues,sequences,seq_name,path):
    hoomd.context.initialize("--mode=cpu");
    hoomd.option.set_notice_level(1)
    hoomd.util.quiet_status()

    seq = sequences.loc[seq_name]

    lj_eps = 4.184*.2
    temp = seq.temp
    ionic_strength = seq.ionic # M
    RT = 8.3145*temp*1e-3

    yukawa_kappa, yukawa_eps, types, pairs, fasta, residues = gen_params(residues,seq,temp,ionic_strength)

    sigmamap = pd.DataFrame((residues.sigmas.values+residues.sigmas.values.reshape(-1,1))/2,
                            index=residues.sigmas.index,columns=residues.sigmas.index)
    lambdamap = pd.DataFrame((residues.lambdas.values+residues.lambdas.values.reshape(-1,1))/2,
                            index=residues.lambdas.index,columns=residues.lambdas.index)

    N_res = seq.N
    L = 200 if N_res > 500 else (N_res-1)*0.38+4
    N_save = 7000 if N_res < 150 else int(np.ceil(3e-4*N_res**2)*1000)
    N_steps = 1010*N_save

    gen_init_top(residues,fasta,path,L)

    snapshot = hoomd.data.make_snapshot(N=N_res,
                                box=hoomd.data.boxdim(Lx=L, Ly=L, Lz=L),
                                particle_types=types,
                                bond_types=['polymer']);

    snapshot.bonds.resize(N_res-1);

    snapshot.particles.position[:] = md.load(path+'/init_top.pdb').xyz
    snapshot.particles.typeid[:] = [types.index(a) for a in fasta]
    snapshot.particles.mass[:] = [residues.loc[a].MW for a in fasta]

    snapshot.bonds.group[:] = [[i,i+1] for i in range(N_res-1)];
    snapshot.bonds.typeid[:] = [0] * (N_res-1)

    hoomd.init.read_snapshot(snapshot);

    hb = hoomd.md.bond.harmonic();
    hb.bond_coeff.set('polymer', k=8033.0, r0=0.38);

    nl = hoomd.md.nlist.cell();

    ah = azplugins.pair.ashbaugh(r_cut=2.0, nlist=nl)
    yukawa = hoomd.md.pair.yukawa(r_cut=4.0, nlist=nl)
    for a,b in pairs:
        ah.pair_coeff.set(a, b, lam=lambdamap.loc[a,b], epsilon=lj_eps, sigma=sigmamap.loc[a,b], r_cut=2.0)
        yukawa.pair_coeff.set(a, b, epsilon=yukawa_eps.loc[a,b], kappa=yukawa_kappa, r_cut=4.)

    ah.set_params(mode='shift')
    yukawa.set_params(mode='shift')
    nl.reset_exclusions(exclusions = ['bond'])

    hoomd.md.integrate.mode_standard(dt=0.005);
    integrator = hoomd.md.integrate.langevin(group=hoomd.group.all(),kT=RT,seed=np.random.randint(100));

    for a in types:
        integrator.set_gamma(a, residues.loc[a].MW/100)

    hoomd.run(10000)

    integrator.disable()

    hoomd.md.integrate.mode_standard(dt=0.01);
    integrator = hoomd.md.integrate.langevin(group=hoomd.group.all(),kT=RT,seed=np.random.randint(100));

    for a in types:
        integrator.set_gamma(a, residues.loc[a].MW/100)

    hoomd.dump.dcd(filename=path+'/traj.dcd', period=N_save, group=hoomd.group.all(), unwrap_full=True, overwrite=True);

    hoomd.run(N_steps)

    gen_xtc(path)

residues = pd.read_csv('residues.csv').set_index('one',drop=False)

sequences = pd.read_csv('sequences.csv',index_col=0)
sequences['N'] = sequences.fasta.apply(len)

t0 = time.time()
simulate(residues,sequences,args.seq_name,args.path)
print('Timing Simulation {:.3f}'.format(time.time()-t0))

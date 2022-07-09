import hoomd
import hoomd.md
from hoomd import azplugins
import time
import itertools
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',const='', type=str)
parser.add_argument('--cutoff',dest='cutoff',type=float)
args = parser.parse_args()

print(hoomd.__file__)

def genParams(r,prot):
    RT = 8.3145*prot.temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(prot.temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.022*1000/RT
    # Calculate the inverse of the Debye length
    yukawa_kappa = np.sqrt(8*np.pi*lB*prot.ionic*6.022/10)
    fasta = prot.fasta.copy()
    # Set the charge on HIS based on the pH of the protein solution? Not needed if pH=7.4
    r.loc['H','q'] = 1. / ( 1 + 10**(prot.pH-6) )
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

def simulate(residues,name,prot,cutoff):
    hoomd.context.initialize("--mode=cpu --nthreads=1");
    hoomd.option.set_notice_level(1)
    hoomd.util.quiet_status()

    lj_eps = 4.184*.2
    RT = 8.3145*prot.temp*1e-3

    yukawa_kappa, yukawa_eps, types, pairs, fasta, residues = genParams(residues,prot)

    sigmamap = pd.DataFrame((residues.sigmas.values+residues.sigmas.values.reshape(-1,1))/2,
                            index=residues.sigmas.index,columns=residues.sigmas.index)
    lambdamap = pd.DataFrame((residues.lambdas.values+residues.lambdas.values.reshape(-1,1))/2,
                            index=residues.lambdas.index,columns=residues.lambdas.index)

    N_res = prot.N
    L = N_res*.38+1
    N_save = 3000 if N_res < 100 else int(np.ceil(3e-4*N_res**2)*1000)
    N_steps = 1600*N_save

    snapshot = hoomd.data.make_snapshot(N=N_res,
                                box=hoomd.data.boxdim(Lx=L, Ly=L, Lz=L),
                                particle_types=types,
                                bond_types=['polymer']);

    snapshot.bonds.resize(N_res-1);

    snapshot.particles.position[:] = [[0,0,(i-N_res/2.)*.38] for i in range(N_res)]
    snapshot.particles.typeid[:] = [types.index(a) for a in fasta]
    snapshot.particles.mass[:] = [residues.loc[a].MW for a in fasta]

    snapshot.bonds.group[:] = [[i,i+1] for i in range(N_res-1)];
    snapshot.bonds.typeid[:] = [0] * (N_res-1)

    hoomd.init.read_snapshot(snapshot);

    hb = hoomd.md.bond.harmonic();
    hb.bond_coeff.set('polymer', k=8033.0, r0=0.38);

    nl = hoomd.md.nlist.cell();

    ah = azplugins.pair.ashbaugh(r_cut=cutoff, nlist=nl)
    yukawa = hoomd.md.pair.yukawa(r_cut=4.0, nlist=nl)
    for a,b in pairs:
        ah.pair_coeff.set(a, b, lam=lambdamap.loc[a,b], epsilon=lj_eps, sigma=sigmamap.loc[a,b], r_cut=cutoff)
        yukawa.pair_coeff.set(a, b, epsilon=yukawa_eps.loc[a,b], kappa=yukawa_kappa, r_cut=4.)

    ah.set_params(mode='shift')
    yukawa.set_params(mode='shift')
    nl.reset_exclusions(exclusions = ['bond'])

    integrator_mode = hoomd.md.integrate.mode_standard(dt=0.005);
    integrator = hoomd.md.integrate.langevin(group=hoomd.group.all(),kT=RT,seed=np.random.randint(100));

    for a in types:
        integrator.set_gamma(a, residues.loc[a].MW/100)

    hoomd.run(10000)

    integrator.disable()

    integrator_mode = hoomd.md.integrate.mode_standard(dt=0.01);
    integrator = hoomd.md.integrate.langevin(group=hoomd.group.all(),kT=RT,seed=np.random.randint(100));

    for a in types:
        integrator.set_gamma(a, residues.loc[a].MW/100)

    hoomd.dump.gsd(filename='{:s}/{:s}.gsd'.format(name,name), period=N_save, group=hoomd.group.all(), overwrite=True);

    hoomd.run(N_steps)

residues = pd.read_csv('residues.csv').set_index('one',drop=False)

proteins = pd.read_pickle('proteins.pkl')

t0 = time.time()
simulate(residues,args.name,proteins.loc[args.name],args.cutoff)
print('Timing Simulation {:.3f}'.format(time.time()-t0))

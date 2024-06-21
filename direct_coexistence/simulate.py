from simtk import openmm, unit
from simtk.openmm import app
from simtk.openmm import XmlSerializer
from analyse import *
import time
import os
import sys
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('--name',nargs='?',const='', type=str)
parser.add_argument('--temp',nargs='?',const='', type=int)
args = parser.parse_args()

def simulate(residues,name,prot,temp):
    residues = residues.set_index('one')

    lj_eps, fasta, types, MWs = genParamsLJ(residues,name,prot)
    yukawa_eps, yukawa_kappa = genParamsDH(residues,name,prot,temp)

    N = len(fasta)

    # set parameters
    L = 15.
    margin = 2
    if N > 350:
        L = 25.
        Lz = 300.
        margin = 8
        Nsteps = int(2e7)
    elif N > 200:
        L = 17.
        Lz = 300.
        margin = 4
        Nsteps = int(6e7)
    else:
        Lz = 10*L

    system = openmm.System()

    # set box vectors
    a = unit.Quantity(np.zeros([3]), unit.nanometers)
    a[0] = L * unit.nanometers
    b = unit.Quantity(np.zeros([3]), unit.nanometers)
    b[1] = L * unit.nanometers
    c = unit.Quantity(np.zeros([3]), unit.nanometers)
    c[2] = Lz * unit.nanometers
    system.setDefaultPeriodicBoxVectors(a, b, c)

    # initial config
    xy = np.empty(0)
    xy = np.append(xy,np.random.rand(2)*(L-margin)-(L-margin)/2).reshape((-1,2))
    for x,y in np.random.rand(1000,2)*(L-margin)-(L-margin)/2:
        x1 = x-L if x>0 else x+L
        y1 = y-L if y>0 else y+L
        if np.all(np.linalg.norm(xy-[x,y],axis=1)>.7):
            if np.all(np.linalg.norm(xy-[x1,y],axis=1)>.7):
                if np.all(np.linalg.norm(xy-[x,y1],axis=1)>.7):
                    xy = np.append(xy,[x,y]).reshape((-1,2))
        if xy.shape[0] == 100:
            break

    n_chains = xy.shape[0]

    pdb_file = name+'/{:d}/top.pdb'.format(temp)

    if os.path.isfile(pdb_file):
        pdb = app.pdbfile.PDBFile(pdb_file)
    else:
        top = md.Topology()
        pos = []
        for x,y in xy:
            chain = top.add_chain()
            pos.append([[x,y,Lz/2+(i-N/2.)*.38] for i in range(N)])
            for resname in fasta:
                residue = top.add_residue(resname, chain)
                top.add_atom(resname, element=md.element.carbon, residue=residue)
            for i in range(chain.n_atoms-1):
                top.add_bond(chain.atom(i),chain.atom(i+1))
        md.Trajectory(np.array(pos).reshape(n_chains*N,3), top, 0, [L,L,Lz], [90,90,90]).save_pdb(pdb_file)
        pdb = app.pdbfile.PDBFile(pdb_file)

    for _ in range(n_chains):
        system.addParticle((residues.loc[prot.fasta[0]].MW+2)*unit.amu)
        for a in prot.fasta[1:-1]:
            system.addParticle(residues.loc[a].MW*unit.amu)
        system.addParticle((residues.loc[prot.fasta[-1]].MW+16)*unit.amu)

    hb = openmm.openmm.HarmonicBondForce()

    energy_expression = f'{lj_eps}*select(step(r-2^(1/6)*s),4*l*((s/r)^12-(s/r)^6-shift),4*((s/r)^12-(s/r)^6-l*shift)+(1-l))'
    ah = openmm.openmm.CustomNonbondedForce(energy_expression+f'; s=0.5*(s1+s2); l=0.5*(l1+l2); shift=(0.5*(s1+s2)/2.0)^12-(0.5*(s1+s2)/2.0)^6')

    ah.addPerParticleParameter('s')
    ah.addPerParticleParameter('l')

    shift = np.exp(-yukawa_kappa*4.0)/4.0
    yu = openmm.openmm.CustomNonbondedForce(f'q*(exp(-{yukawa_kappa}*r)/r-{shift}); q=q1*q2')
    yu.addPerParticleParameter('q')

    for j in range(n_chains):
        begin = j*N
        end = j*N+N

        for a,e in zip(prot.fasta,yukawa_eps):
            yu.addParticle([e*unit.nanometer*unit.kilojoules_per_mole])
            ah.addParticle([residues.loc[a].sigmas*unit.nanometer, residues.loc[a].lambdas*unit.dimensionless])

        for i in range(begin,end-1):
            hb.addBond(i, i+1, 0.38*unit.nanometer, 8033.0*unit.kilojoules_per_mole/(unit.nanometer**2))
            yu.addExclusion(i, i+1)
            ah.addExclusion(i, i+1)

    yu.setForceGroup(0)
    ah.setForceGroup(1)
    yu.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    ah.setNonbondedMethod(openmm.openmm.CustomNonbondedForce.CutoffPeriodic)
    hb.setUsesPeriodicBoundaryConditions(True)
    yu.setCutoffDistance(4.0*unit.nanometer)
    ah.setCutoffDistance(2.0*unit.nanometer)

    system.addForce(hb)
    system.addForce(yu)
    system.addForce(ah)

    print(ah.usesPeriodicBoundaryConditions())
    print(yu.usesPeriodicBoundaryConditions())
    print(hb.usesPeriodicBoundaryConditions())

    integrator = openmm.openmm.LangevinIntegrator(temp*unit.kelvin,0.01/unit.picosecond,0.01*unit.picosecond)

    print(integrator.getFriction(),integrator.getTemperature())

    platform = openmm.Platform.getPlatformByName('CUDA')

    simulation = app.simulation.Simulation(pdb.topology, system, integrator, platform, dict(CudaPrecision='mixed'))

    check_point = name+'/{:d}/restart.chk'.format(temp)

    if os.path.isfile(check_point):
        print('Reading check point file')
        simulation.loadCheckpoint(check_point)
        simulation.reporters.append(app.dcdreporter.DCDReporter(name+'/{:d}/{:s}.dcd'.format(temp,name),int(1e5),append=True))
    else:
        simulation.context.setPositions(pdb.positions)
        simulation.minimizeEnergy()
        simulation.reporters.append(app.dcdreporter.DCDReporter(name+'/{:d}/{:s}.dcd'.format(temp,name),int(1e5)))

    simulation.reporters.append(app.statedatareporter.StateDataReporter('{:s}_{:d}.log'.format(name,temp),1000000,
             step=True,speed=True,elapsedTime=True,separator='\t'))

    simulation.runForClockTime(20*unit.hour, checkpointFile=check_point, checkpointInterval=2*unit.hour)

    simulation.saveCheckpoint(check_point)

    center_slab(name)

residues = pd.read_csv('residues.csv').set_index('three',drop=False)
sequences = pd.read_csv('sequences.csv',index_col=0)
print(args.name,args.temp)
t0 = time.time()
simulate(residues,args.name,sequences.loc[args.name],args.temp)
print('Timing {:.3f}'.format(time.time()-t0))

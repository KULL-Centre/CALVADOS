import pytest
import numpy as np
import os
import pandas as pd
from calvados.cfg import Config, Job, Components
from calvados import sim
import subprocess
import numpy as np
import mdtraj as md

# Ashbaugh-Hatch potential
HALR = lambda r,s,l : 4*0.8368*l*((s/r)**12-(s/r)**6)
HASR = lambda r,s,l : 4*0.8368*((s/r)**12-(s/r)**6)+0.8368*(1-l)
HA = lambda r,s,l : np.where(r<2**(1/6)*s, HASR(r,s,l), HALR(r,s,l))
HASP = lambda r,s,l,rc : np.where(r<rc, HA(r,s,l)-HA(rc,s,l), 0)

# Debye-Hückel potential
DH = lambda r,yukawa_eps,lD : yukawa_eps*np.exp(-r/lD)/r
DHSP = lambda r,yukawa_eps,lD,rc : np.where(r<rc, DH(r,yukawa_eps,lD)-DH(rc,yukawa_eps,lD), 0)

@pytest.mark.parametrize(
    ("resname1", "resname2"),
    [
        ("Y", "W"),
        ("R", "W"),
        ("E", "D"),
        ("E", "W"),
        ("E", "R"),
    ],
)

def test_ah_dh_potentials(resname1,resname2):

    cwd = os.getcwd()

    sysname = f'{resname1:s}_{resname2:s}'

    # set the side length of the cubic box
    L = 8

    # set the temperature
    temp = 298

    # set ionic strength
    ionic = 0.15

    # set the saving interval (number of integration steps)
    N_save = 10

    # set final number of frames to save
    N_frames = 10000

    residues_file = f'{cwd}/tests/data/residues_CALVADOS2.csv'
    fasta_file = f'{cwd}/tests/data/fastalib.fasta'

    config = Config(
    # GENERAL
    sysname = sysname, # name of simulation system
    box = [L, L, L], # nm
    temp = temp, # K
    ionic = ionic, # molar
    pH = 7.0,
    topol = 'grid',

    # RUNTIME SETTINGS
    wfreq = N_save, # dcd writing interval, 1 = 10 fs
    logfreq = N_save, # log file writing interval, 1 = 10 fs
    steps = N_frames*N_save, # number of simulation steps
    platform = 'CPU', # or CUDA
    restart = None,
    verbose = False,
    report_potential_energy = True,
    random_number_seed = 12345,
    )

    # PATH
    path = f'{cwd}/tests/data/{sysname:s}'

    subprocess.run(f'mkdir -p {path}',shell=True)

    config.write(path,name='config.yaml')

    components = Components(
    # Defaults
    molecule_type = 'protein',
    nmol = 1, # number of molecules
    fresidues = residues_file, # residue definitions
    ffasta = fasta_file, # domain definitions (harmonic restraints)
    )
    components.add(name=resname1, restraint=False, charge_termini='none')
    components.add(name=resname2, restraint=False, charge_termini='none')

    components.write(path,name='components.yaml')

    sim.run(path=path,fconfig='config.yaml',fcomponents='components.yaml')

    t = md.load(f'{cwd}/tests/data/{sysname:s}/{sysname:s}.dcd',
                top=f'{cwd}/tests/data/{sysname:s}/top.pdb')

    # compute distance between beads
    dist = md.compute_distances(traj=t,atom_pairs=[[0,1]])[:,0]

    # load potential energy
    u = np.loadtxt(f'{cwd}/tests/data/{sysname:s}/{sysname:s}.log',usecols=(1))

    # calculate potential energy based on bead-bead distance
    df_residues = pd.read_csv(residues_file,index_col=0)
    sigma_ij = 0.5*(df_residues.loc[resname1].sigmas + df_residues.loc[resname2].sigmas)
    lambda_ij = 0.5*(df_residues.loc[resname1].lambdas + df_residues.loc[resname2].lambdas)
    u_ah = HASP(dist,sigma_ij,lambda_ij,2)

    RT = 8.3145*temp*1e-3
    fepsw = lambda T : 5321/T+233.76-0.9297*T+0.1417*1e-2*T*T-0.8292*1e-6*T**3
    epsw = fepsw(temp)
    lB = 1.6021766**2/(4*np.pi*8.854188*epsw)*6.02214076*1000/RT
    yukawa_eps = df_residues.loc[resname1].q*df_residues.loc[resname2].q*lB*RT
    lD = 1. / np.sqrt(8*np.pi*lB*ionic*6.02214076/10)

    u_dh = DHSP(dist,yukawa_eps,lD,4)
    u_calc = u_ah + u_dh
    abs_err = np.abs(u_calc-u)
    u_abs = np.abs(u)
    print('Distance of max abs error:',dist[abs_err.argmax()])
    print('Distance of max abs error / sigma_ij:',dist[abs_err.argmax()]/sigma_ij)
    print('Max Relative Error:',(abs_err[u_abs>0]/u_abs[u_abs>0]).max())
    assert np.allclose(u,u_calc,rtol=1e-3,atol=1e-8)


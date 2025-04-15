import numpy as np

from MDAnalysis import Universe

import random

from re import findall

from Bio import SeqIO, SeqUtils
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

import os

from localcider.sequenceParameters import SequenceParameters
import tqdm as tqdm
import warnings

from joblib import load

import numba as nb
from scipy.integrate import quad

from calvados import analysis, interactions

### SEQUENCE INPUT / OUTPUT
def read_fasta(ffasta):
    records = SeqIO.to_dict(SeqIO.parse(ffasta, "fasta"))
    return records

def seq_from_pdb(pdb,selection='all',fmt='string'):
    """ Generate fasta from pdb entries """

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        u = Universe(pdb)

    # we do not assume residues in the PDB file are numbered from 1
    n_termini = [0]
    c_termini = [len(u.atoms.segments[0].residues)-1]
    for segment in u.atoms.segments[1:]:
        n_termini.append(c_termini[-1]+1)
        c_termini.append(c_termini[-1]+len(segment.residues))

    ag = u.select_atoms(selection)
    if fmt == 'string':
        fastapdb = ""
    elif fmt == 'list':
        fastapdb = []
    else:
        raise
    res3 = ag.residues.resnames
    for res in res3:
        if len(res) == 3:
            res1 = SeqUtils.seq1(res)
        else:
            res1 = res
        if res1 == "":
            res1 = "X"
        fastapdb += res1
    return fastapdb, n_termini, c_termini

def write_fasta(new_records,fout):
    """ seqs: list of sequences """
    if os.path.isfile(fout):
        records = []
        ids = []
        for record in SeqIO.parse(fout, "fasta"):
            records.append(record)
            ids.append(record.id)
        for record in new_records:
            if record.id not in ids: # avoid duplicates
                records.append(record)
        SeqIO.write(records, fout, "fasta")
    else:
        SeqIO.write(new_records, fout, "fasta")

def record_from_seq(seq,name):
    record = SeqRecord(
        Seq(seq),
        id=name,
        name='',
        description=''
    )
    return(record)

### SEQUENCE ANALYSIS

def get_qs(seq,flexhis=False,pH=7,residues=[]):
    """ charges and absolute charges vs. residues """
    qcoeff = 1.
    qs = []
    # scaled charges
    if len(residues) == 0:
        qs, qs_abs = get_qs_fast(seq,flexhis=flexhis,pH=pH)
    else:
        for s in seq:
            if flexhis and s == 'H':
                q = qcoeff / ( 1 + 10**(pH-6) )
            else:
                q = residues.loc[s].q
            qs.append(q)
        qs = np.asarray(qs).astype(float)
        qs_abs = np.abs(qs)
    return qs, qs_abs

@nb.jit(nopython=True)
def get_qs_fast(seq,flexhis=False,pH=7.):
    """ charges and absolute charges vs. residues """
    qs = np.zeros(len(seq))
    qs_abs = np.zeros(len(seq))

    # loop through sequence
    for idx in range(len(seq)):
        if seq[idx] in ['R','K']:
            qs[idx] = 1.
            qs_abs[idx] = 1.
        elif seq[idx] in ['E','D','p']:
            qs[idx] = -1.
            qs_abs[idx] = 1.
        elif (seq[idx] == 'H') and flexhis:
            qs[idx] = 1. / ( 1 + 10**(pH-6) )
            qs_abs[idx] = 1. / ( 1 + 10**(pH-6) )
        else:
            qs[idx] = 0.
            qs_abs[idx] = 0.
    return qs, qs_abs

def patch_terminal_qs(qs,n_termini,c_termini,loc='both'):
    qsnew = qs.copy()
    qcoeff = 1.

    if loc in ['N','both']:
        qsnew[n_termini] += qcoeff
    if loc in ['C','both']:
        qsnew[c_termini] -= qcoeff
    return qsnew

def seq_dipole(seq):
    """ 1D charge dipole along seq """
    # print(seq)
    qs, qs_abs = get_qs(seq)
    com = seq_com(qs_abs)
    dip = 0.
    # ct = 0
    for idx,q in enumerate(qs):
        dip += (com - idx) * q # positive if positive towards Nterm
    return com, dip

def seq_com(qs_abs):
    """ center of charges """
    com = 0.
    ct = 0.
    for idx,q_abs in enumerate(qs_abs):
        ct += int(q_abs)
        com += int(q_abs) * idx
    if ct > 0:
        com /= ct
    else:
        com = len(qs_abs) // 2
    return com

# @nb.jit(nopython=True)
# def calc_SCD(seq,charge_termini=False):
#     """ Sequence charge decoration, eq. 14 in Sawle & Ghosh, JCP 2015 """
#     qs, _ = get_qs_fast(seq)
#     if charge_termini:
#         qs[0] = qs[0] + 1.
#         qs[-1] = qs[-1] - 1.
#     N = len(seq)
#     scd = 0.
#     for idx in range(1,N):
#         for jdx in range(0,idx):
#             s = qs[idx] * qs[jdx] * (idx - jdx)**0.5
#             scd = scd + s
#     scd = scd / N
#     return scd

@nb.jit(nopython=True)
def calc_SCD(qs):
    """ Sequence charge decoration, eq. 14 in Sawle & Ghosh, JCP 2015 """
    N = len(qs)
    scd = 0.
    for idx in range(1,N):
        for jdx in range(0,idx):
            s = qs[idx] * qs[jdx] * (idx - jdx)**0.5
            scd = scd + s
    scd = scd / N
    return scd

# @nb.jit(nopython=True)
def calc_SHD(seq,lambda_map,beta=-1.):
    """ Sequence hydropathy decoration, eq. 4 in Zheng et al., JPC Letters 2020"""
    N = len(seq)
    shd = 0.
    # lambdas = residues.lambdas[list(seq)].to_numpy()
    # lambda_map = np.add.outer(lambdas,lambdas)

    for idx in range(0, N-1):
        seqi = seq[idx]
        for jdx in range(idx+1, N):
            seqj = seq[jdx]
            s = lambda_map[(seqi,seqj)] * (jdx - idx)**beta
            # s = lambda_map[idx,jdx] * (jdx - idx)**beta
            shd = shd + s
    shd = shd / N
    return shd

def mean_lambda(seq,residues):
    """ Mean hydropathy """
    lambdas_sum = 0.
    for idx, x in enumerate(seq):
        lambdas_sum += residues.lambdas[x]
    lambdas_mean = lambdas_sum / len(seq)
    return  lambdas_mean

def calc_aromatics(seq):
    """ Fraction of aromatics """
    seq = str(seq)
    N = len(seq)
    rY = len(findall('Y',seq)) / N
    rF = len(findall('F',seq)) / N
    rW = len(findall('W',seq)) / N
    return rY, rF, rW

def calc_mw(fasta,residues=[]):
    seq = "".join(fasta)
    if len(residues) > 0:
        mw = 0.
        for s in seq:
            m = residues.loc[s,'MW']
            mw += m
    else:
        mw = SeqUtils.molecular_weight(seq,seq_type='protein')
    return mw

def calc_kappa(seq):
    seq = "".join(seq)
    SeqOb = SequenceParameters(seq)
    k = SeqOb.get_kappa()
    return k

### SEQUENCE MANIPULATION
def shuffle_str(seq):
    l = list(seq)
    random.shuffle(l)
    seq_shuffled = "".join(l)
    return(seq_shuffled)

def construct_maxdipseq(seq):
    """ sequence permutation with largest dipole """
    seqpos, seqneg, seqneu = split_seq(seq)
    seqout = seqpos + seqneu + seqneg
    return seqout

def split_seq(seq):
    """ split sequence in positive, negative, neutral residues """
    seqpos = []
    seqneg = []
    seqneu = []
    for s in seq:
        if s in ['K','R']:
            seqpos.append(s)
        elif s in ['D','E']:
            seqneg.append(s)
        else:
            seqneu.append(s)
    seqpos = shuffle_str(seqpos)
    seqneg = shuffle_str(seqneg)
    seqneu = shuffle_str(seqneu)
    return seqpos, seqneg, seqneu

def single_swap(seq):
    seq = list(seq)
    N = len(seq)
    while True:
        i = random.choice(range(N))
        j = random.choice(range(N))
        if i != j and seq[i] != seq[j]:
            break
    if (seq[i] in ['K','R','D','E']) or (seq[j] in ['K','R','D','E']):
        charge_swap = True
    else:
        charge_swap = False
    seq = swap_pos(seq,i,j)
    return seq, charge_swap

def k_energy(k,k_target,a_kappa=1.):
    # k = calc_kappa(seq)
    if k > k_target:
        u = a_kappa*(k-k_target)**2
    else:
        u = a_kappa*(k-k_target)**2
    return u

def dip_energy(seq,dip_target,a_dip=1.,dipmax=0.):
    com, dip = seq_dipole(seq)
    dip_red = dip / dipmax #dip_per_charge/len(seq)
    u = a_dip * (dip_red - dip_target)**2
    return u

def metropolis(u0,u1,a=100.):
    if u1 < u0:
        return True
    else:
        x = np.random.random()
        d = np.exp(a*(u0-u1))
        # print(x,d)
        if d > x:
            return True
        else:
            return False

def swap_pos(seq,i,j):
    seq = list(seq)
    seq[i], seq[j] = seq[j], seq[i]
    seq = "".join(seq)
    return seq


# @nb.jit(nopython=True)
def calc_qpatch(seq,charge_termini=False,beta=0.5):
    """ Patchiness """
    qs, _ = get_qs(seq)
    if charge_termini:
        qs[0] += 1.
        qs[-1] -= 1.
    N = len(seq)

    xs = np.arange(N)

    qmap = np.multiply.outer(qs,qs)
    r = (np.abs(np.subtract.outer(xs,xs))+1)**beta

    U = get_U(N,qmap,r)
    U /= (N*(N-1) / 2 + N)
    return U

@nb.jit(nopython=True)
def get_U(N,qmap,r,cutoff=100):
    U = 0.
    for idx in range(0,N):
        for kdx in range(idx,N):
            for jdx in range(max(0,kdx-cutoff),min(kdx+cutoff+1,N)):
                u = qmap[idx,jdx] / r[jdx,kdx]
                U = U + u
    return U

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

## ah pairs
def ah_pairs(seq,residues,rc=2.,eps=0.2*4.184,r0=0.6,beta=0.5,maxdist=100):

    seq = list(seq)
    N = len(seq)

    # distances
    xs = np.arange(N+1)
    rs = r0*xs**beta # nm

    # sigma, lambda maps
    sig_map, l_map = make_sig_lambda_map(seq,residues)

    U = ikj_loop_ah(N,rs,sig_map,l_map,eps,rc,maxdist)
    return U

def make_sig_lambda_map(seq,residues):
    seq = list(seq)

    sigs = residues.loc[seq,'sigmas'].to_numpy()
    ls = residues.loc[seq,'lambdas'].to_numpy()

    sig_map = np.add.outer(sigs,sigs) / 2.
    l_map = np.add.outer(ls,ls) / 2.
    return sig_map, l_map

@nb.jit(nopython=True)
def ikj_loop_ah(N,rs,sig_map,l_map,eps,rc,maxdist):
    U = 0
    for i in range(N):
        for k in range(N):
            for j in range(max(0,k-maxdist),min(N,k+maxdist+1)):
                r = rs[abs(k-j)+1]
                sig = sig_map[i,j]
                l = l_map[i,j]
                u = ah_potential(r,sig,eps,l,rc)
                U = U + u
    U = U / (N*(N-1)/2 + N)
    return U

## q pairs
def q_pairs(seq,residues,r0=0.6,beta=0.5,maxdist=100,temp=293,ionic=0.15,rc_yu=4.0):

    seq = list(seq)
    N = len(seq)

    # distances
    xs = np.arange(N+1)
    rs = r0*xs**beta # nm

    # q maps
    eps_yu, k_yu = interactions.genParamsDH(temp,ionic)
    q_map = make_q_map(seq,residues) * eps_yu**2

    U = ikj_loop_q(N,rs,q_map,k_yu,rc_yu,maxdist)
    return U

def make_q_map(seq,residues):
    seq = list(seq)
    qs = residues.loc[seq,'q'].to_numpy()
    q_map = np.multiply.outer(qs,qs)
    return q_map

@nb.jit(nopython=True)
def ikj_loop_q(N,rs,q_map,k_yu,rc_yu,maxdist):
    U = 0
    for i in range(N):
        for k in range(N):
            for j in range(max(0,k-maxdist),min(N,k+maxdist+1)):
                r = rs[abs(k-j)+1]
                q = q_map[i,j] # includes eps_yu**2
                if abs(q) > 0.:
                    u = analysis.yukawa_potential(r,q,k_yu,rc_yu=rc_yu)
                    U = U + u
    U = U / (N*(N-1)/2 + N)
    return U

def ah_single(seq,residues,rc=2.,eps=0.2*4.184,r0=0.6,beta=0.5):
    """ Sequence hydropathy decoration, eq. 4 in Zheng et al., JPC Letters 2020"""

    seq = list(seq)
    N = len(seq)

    # distances
    xs = np.arange(N+1)
    rs = r0*xs**beta # nm

    # sigma, lambda maps
    sig_map, l_map = make_sig_lambda_map(seq,residues)
    U = ij_loop_ah(N,rs,sig_map,l_map,eps,rc)
    return U

@nb.jit(nopython=True)
def ij_loop_ah(N,rs,sig_map,l_map,eps,rc):
    U = 0
    for i in range(N-2):
        for j in (i+2,N):
            r = rs[j-i]
            sig = sig_map[i,j]
            l = l_map[i,j]
            u = ah_potential(r,sig,eps,l,rc)
            U = U + u
    U = U / N
    return U

def ah_scaled(r,sig,eps,l,rc):
    ah = ah_potential(r,sig,eps,l,rc)
    ahs = ah*4*np.pi*r**2
    return ahs

def make_ah_intgrl_map(residues,rc=2.0,eps = 0.2 * 4.184):
    ah_intgrl_map = {}
    for key0, val0 in residues.iterrows():
        sig0, l0 = val0['sigmas'], val0['lambdas']
        for key1, val1 in residues.iterrows():
            sig1, l1 = val1['sigmas'], val1['lambdas']
            sig, l = 0.5*(sig0+sig1), 0.5*(l0+l1)
            res = quad(lambda r: ah_scaled(r,sig,eps,l,rc), 2**(1./6.)*sig, rc)
            ah_intgrl_map[(key0,key1)] = res[0]
            ah_intgrl_map[(key1,key0)] = res[0]
    return ah_intgrl_map

def make_lambda_map(residues):
    lambda_map = {}
    for key0, val0 in residues.iterrows():
        l0 = val0['lambdas']
        for key1, val1 in residues.iterrows():
            l1 = val1['lambdas']
            l = l0+l1
            lambda_map[(key0,key1)] = l
            lambda_map[(key1,key0)] = l
    return lambda_map

def calc_ah_ij(seq,ah_intgrl_map):
    U = 0.
    seq = list(seq)
    N = len(seq)
    for idx in range(N):
        seqi = seq[idx]
        for jdx in range(idx,N):
            seqj = seq[jdx]
            ahi = ah_intgrl_map[(seqi,seqj)]
            U += ahi
    U /= (N * (N-1) / 2. + N)
    return U

def yu_scaled(r,q,k_yu,rc_yu=4.):
    yu = analysis.yukawa_potential(r,q,k_yu,rc_yu=rc_yu)
    yus = yu*4*np.pi*r**2
    return yus

def make_q_intgrl_map(residues,temp=293.,ionic=0.15,rc_yu=4.):
    q_intgrl_map = {}
    eps_yu, k_yu = interactions.genParamsDH(temp,ionic)
    for key0, val0 in residues.iterrows():
        q0, sig0 = val0['q'], val0['sigmas']
        for key1, val1 in residues.iterrows():
            q1, sig1 = val1['q'], val1['sigmas']
            q = q0*q1 * eps_yu**2
            sig = 0.5*(sig0+sig1)
            res = quad(lambda r: yu_scaled(r,q,k_yu,rc_yu=rc_yu), 2**(1./6.)*sig, rc_yu)
            q_intgrl_map[(key0,key1)] = res[0]
            q_intgrl_map[(key1,key0)] = res[0]
    return q_intgrl_map

def calc_q_ij(seq,q_intgrl_map):
    U = 0.
    seq = list(seq)
    N = len(seq)
    for idx in range(N):
        seqi = seq[idx]
        for jdx in range(idx,N):
            seqj = seq[jdx]
            u = q_intgrl_map[(seqi,seqj)]
            U += u
    U /= (N * (N-1) / 2. + N)
    return U

############ FAST KAPPA ################

@nb.jit(nopython=True)
def check_dmax(seq,dmax,seqmax):
    qs, _ = get_qs_fast(seq)
    # qs, _ = get_qs(seq)
    d = calc_delta(qs)
    if d > dmax:
        return seq, d
    else:
        return seqmax, dmax

@nb.jit(nopython=True)
def calc_case0(seqpos,seqneg,seqneu):
    seqmax = ''
    dmax = 0.
    N = len(seqpos) + len(seqneg) + len(seqneu)
    if len(seqpos) == 0:
        seqcharge = seqneg[:]
    elif len(seqneg) == 0:
        seqcharge = seqpos[:]
    if len(seqneu) > len(seqcharge):
        for pos in range(0, N - len(seqcharge) + 1):
            seqout = ''
            seqout += seqneu[:pos]
            seqout += seqcharge
            seqout += seqneu[pos:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    else:
        for pos in range(0, N - len(seqneu) + 1):
            seqout = ''
            seqout += seqcharge[:pos]
            seqout += seqneu
            seqout += seqcharge[pos:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    return seqmax

@nb.jit(nopython=True)
def calc_case1(seqpos,seqneg,seqneu):
    seqmax = ''
    dmax = 0.
    N = len(seqpos) + len(seqneg) + len(seqneu)
    if len(seqpos) > len(seqneg):
        for pos in range(0, N - len(seqneg) + 1):
            seqout = ''
            seqout += seqpos[:pos]
            seqout += seqneg
            seqout += seqpos[pos:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    else:
        for neg in range(0, N - len(seqpos) + 1):
            seqout = ''
            seqout += seqneg[:neg]
            seqout += seqpos
            seqout += seqneg[neg:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    return seqmax

@nb.jit(nopython=True)
def calc_case2(seqpos,seqneg,seqneu):
    seqmax = ''
    dmax = 0.
    for startNeuts in range(0, 7):
        for endNeuts in range(0, 7):
            startBlock = seqneu[:startNeuts]
            endBlock = seqneu[startNeuts:startNeuts+endNeuts]
            midBlock = seqneu[startNeuts+endNeuts:]

            seqout = ''
            seqout += startBlock
            seqout += seqpos
            seqout += midBlock
            seqout += seqneg
            seqout += endBlock
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    return seqmax

@nb.jit(nopython=True)
def calc_case3(seqpos,seqneg,seqneu):
    seqmax = ''
    dmax = 0.
    for midNeuts in range(0, len(seqneu)+1):
        midBlock = seqneu[:midNeuts]
        for startNeuts in range(0, len(seqneu) - midNeuts + 1):
            startBlock = seqneu[midNeuts:midNeuts+startNeuts]
            seqout = ''
            seqout += startBlock
            seqout += seqpos
            seqout += midBlock
            seqout += seqneg
            seqout += seqneu[midNeuts+startNeuts:]
            seqmax, dmax = check_dmax(seqout,dmax,seqmax)
    return seqmax

def construct_deltamax(seq):
    seqpos, seqneg, seqneu = split_seq(seq)

    if (len(seqpos) == 0) or (len(seqneg) == 0):
        seqmax = calc_case0(seqpos,seqneg,seqneu)
    elif len(seqneu) == 0:
        seqmax = calc_case1(seqpos,seqneg,seqneu)
    elif len(seqneu) >= 18:
        seqmax = calc_case2(seqpos,seqneg,seqneu)
    else:
        seqmax = calc_case3(seqpos,seqneg,seqneu)
    return seqmax

def calc_kappa_manual(seq,residues=[]):
    qs, qs_abs = get_qs(seq,residues=residues,flexhis=False)
    if np.sum(qs_abs) == 0:
        return -1
    else:
        seqpos, seqneg, seqneu = split_seq(seq)
        if (len(seqneu) == 0):
            if (len(seqneg) == 0) or (len(seqpos) == 0):
                return -1

    delta = calc_delta(qs)

    seq_max = construct_deltamax(seq)
    qs_max, _ = get_qs_fast(seq_max)
    delta_max = calc_delta(qs_max)

    kappa = delta / delta_max
    return kappa

@nb.jit(nopython=True)
def calc_delta(qs):
    d5 = calc_delta_form(qs,window=5)
    d6 = calc_delta_form(qs,window=6)
    return (d5 + d6) / 2.

@nb.jit(nopython=True)
def calc_delta_form(qs,window=5):
    sig_m = calc_sigma(qs)

    nw = len(qs)-window + 1
    sigs = np.zeros((nw))

    for idx in range(0,nw):
        q_window = qs[idx:idx+window]
        sigs[idx] = calc_sigma(q_window)
    delta = np.sum((sigs-sig_m)**2) / nw
    return delta

@nb.jit(nopython=True)
def frac_charges(qs):
    N = len(qs)
    fpos = 0.
    fneg = 0.
    for idx in range(N):
        if qs[idx] >= 1:
            fpos = fpos + 1.
        elif qs[idx] <= -1:
            fneg = fneg + 1.
    fpos = fpos / N
    fneg = fneg / N
    return fpos, fneg

@nb.jit(nopython=True)
def calc_sigma(qs):
    fpos, fneg = frac_charges(qs)
    ncpr = fpos-fneg
    fcr = fpos+fneg
    if fcr == 0:
        return 0.
    else:
        return ncpr**2 / fcr

class SeqFeatures:
    def __init__(self,seq,residues=None,charge_termini=False,calc_dip=False,
    nu_file=None,ah_intgrl_map=None,lambda_map=None,flexhis=False,pH=7.):
        self.seq = seq
        self.N = len(seq)
        if flexhis:
            self.qs, self.qs_abs = get_qs(seq,flexhis=flexhis,pH=pH,residues=residues)
        else:
            self.qs, self.qs_abs = get_qs(seq,residues=residues)
        if charge_termini:
            self.qs[0] += 1.
            self.qs[-1] -= 1.
        self.charge = np.sum(self.qs)
        self.fpos, self.fneg = frac_charges(self.qs)
        self.ncpr = self.charge / self.N
        self.fcr = self.fpos+self.fneg
        self.scd = calc_SCD(self.qs)
        self.rY, self.rF, self.rW = calc_aromatics(seq)
        self.faro = self.rY + self.rF + self.rW

        if calc_dip:
            self.com, self.dip = seq_dipole(seq)
            self.seqmax = construct_maxdipseq(seq)
            self.commax, self.dipmax = seq_dipole(self.seqmax)
            self.dipred = self.dip / self.dipmax

        if residues is not None:
            self.lambdas_mean = mean_lambda(seq,residues)
            self.mean_lambda = self.lambdas_mean
            if lambda_map is None:
                lambda_map = make_lambda_map(residues)
            self.shd = calc_SHD(seq,lambda_map,beta=-1.)
            self.mw = calc_mw(seq,residues=residues)
            if ah_intgrl_map is None:
                ah_intgrl_map = make_ah_intgrl_map(residues)
            self.ah_ij = calc_ah_ij(seq,ah_intgrl_map)

            # q_intgrl_map = make_q_intgrl_map(residues)
            # self.q_ij = calc_q_ij(seq,q_intgrl_map)
            
        if nu_file is not None:
            self.kappa = calc_kappa_manual(seq,residues=residues)
            if self.kappa == -1: # no charges
                self.kappa = 0.
            if flexhis:
                self.qs_noflex, _ = get_qs(seq,residues=residues,flexhis=False)
                if charge_termini:
                    self.qs_noflex[0] += 1.
                    self.qs_noflex[-1] -= 1.
                self.scd_noflex = calc_SCD(self.qs_noflex)
            else:
                self.scd_noflex = self.scd
            feats_for_nu = [self.scd_noflex, self.shd, self.kappa, self.fcr, self.mean_lambda]
            model_nu = load(nu_file)
            X_nu = np.reshape(np.array(feats_for_nu),(1,-1))
            self.nu_svr = model_nu.predict(X_nu)[0]
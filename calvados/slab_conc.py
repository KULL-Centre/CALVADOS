import mdtraj as md
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from numba import jit
import string
from scipy.ndimage import gaussian_filter1d
from mdtraj import element
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import os
from scipy.optimize import least_squares
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import curve_fit
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cycler import cycler
from matplotlib.colors import LogNorm
import warnings
import itertools
warnings.filterwarnings('ignore')
import MDAnalysis as mda
import MDAnalysis.analysis.msd as msd
from statsmodels.tsa.stattools import acf
import sys
from calvados import analysis
import sys
from pathlib import Path
PACKAGEDIR = Path(__file__).parent.absolute()
sys.path.append(f'{str(PACKAGEDIR):s}/BLOCKING')
from main import BlockAnalysis

def calcProfile(seq,name,T,L,value,error,tmin=1200,tmax=None,fbase='.',
    plot=False,pairs=False,X=None,pden=3.,pdil=6.):
    if not pairs:
        X = name
    h = np.load(f'{fbase}/{name}/{T}/{X}_{T}.npy')
    # h = np.load(f'{fbase}/{name}/{T:d}/{X}.npy')
    # print(h.shape)
    N = len(seq)
    conv = 100/6.022/N/L/L*1e3
    h = h[tmin:tmax]*conv # corresponds to (binned) concentration in mM
    lz = h.shape[1]+1
    edges = np.arange(-lz/2.,lz/2.,1)/10
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d) # hyperbolic function, parameters correspond to csat etc.
    residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )
    hm = np.mean(h,axis=0)
    z1 = z[z>0]
    h1 = hm[z>0]
    z2 = z[z<0]
    h2 = hm[z<0]
    p0=[1,1,1,1]
    res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[100]*4)) # fit to hyperbolic function
    res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[100]*4))

    cutoffs1 = [res1.x[2]-pden*res1.x[3],-res2.x[2]+pden*res2.x[3]] # position of interface - half width
    cutoffs2 = [res1.x[2]+pdil*res1.x[3],-res2.x[2]-pdil*res2.x[3]] # get far enough from interface for dilute phase calculation

    bool1 = np.logical_and(z<cutoffs1[0],z>cutoffs1[1])
    bool2 = np.logical_or(z>cutoffs2[0],z<cutoffs2[1])

    dilarray = np.apply_along_axis(lambda a: a[bool2].mean(), 1, h) # concentration in range [bool2]
    denarray = np.apply_along_axis(lambda a: a[bool1].mean(), 1, h)

    if (np.abs(cutoffs2[1]/cutoffs2[0]) > 2) or (np.abs(cutoffs2[1]/cutoffs2[0]) < 0.5): # ratio between right and left should be close to 1
        print('NOT CONVERGED',name,cutoffs1,cutoffs2)
        print(res1.x,res2.x)

    if plot:
        N = 10
        fig, ax = plt.subplots(1,2,figsize=(8,4))
        ax[0].plot(z,hm)
        for c1,c2 in zip(cutoffs1,cutoffs2):
            ax[0].axvline(c1,color='gray')
            ax[0].axvline(c2,color='black')
        ax[0].set(xlabel='z [nm]', ylabel='Concentration [mM]')
        ax[0].set(yscale='log')
        ax[0].set(title=name)
        xs = np.arange(len(h))
        dilrunavg = np.convolve(dilarray, np.ones(N)/N, mode='same')
        ax[1].plot(xs,dilarray)
        # ax[1].plot(xs,dilrunavg)
        ax[1].set(xlabel='Timestep',ylabel='cdil [mM]')
        fig.tight_layout()

    cdil = hm[bool2].mean() # average concentration
    cden = hm[bool1].mean()

    block_dil = BlockAnalysis(dilarray)
    block_den = BlockAnalysis(denarray)
    block_dil.SEM()
    block_den.SEM()

    if pairs:
        value.loc[name,f'{X}_dil'] = cdil # block_dil.av
        value.loc[name,f'{X}_den'] = cden # block_den.av

        error.loc[name,f'{X}_dil'] = block_dil.sem
        error.loc[name,f'{X}_den'] = block_den.sem
    else:
        value.loc[name,'{:d}_dil'.format(T)] = cdil # block_dil.av
        value.loc[name,'{:d}_den'.format(T)] = cden # cilblock_den.av

        error.loc[name,'{:d}_dil'.format(T)] = block_dil.sem
        error.loc[name,'{:d}_den'.format(T)] = block_den.sem

    return(value, error)

def calcProfile_toref(name,T,L,seqX,seqref,value,error,tmin=1200,tmax=None,fbase='.',
    plot=False,X=None,ref=None,pden=3.,pdil=6.):

    # h = np.load(f'{fbase}/{m}/{T:d}/{X}_{T:d}.npy')
    href = np.load(f'{fbase}/{name}/{T}/{ref}.npy')
    hX = np.load(f'{fbase}/{name}/{T}/{X}.npy')

    # print(h.shape)
    Nref = len(seqref)
    convref = 100/6.022/Nref/L/L*1e3

    NX = len(seqX)
    convX = 100/6.022/NX/L/L*1e3

    href = href[tmin:tmax]*convref # corresponds to (binned) concentration in mM
    hX = hX[tmin:tmax]*convX

    lz = href.shape[1]+1
    edges = np.arange(-lz/2.,lz/2.,1)/10
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz

    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d) # hyperbolic function, parameters correspond to csat etc.
    residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )

    hmref = np.mean(href,axis=0)
    hmX = np.mean(hX,axis=0)
    z1 = z[z>0]
    h1 = hmref[z>0]
    z2 = z[z<0]
    h2 = hmref[z<0]
    p0=[1,1,1,1]
    res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[100]*4)) # fit to hyperbolic function
    res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[100]*4))

    cutoffs1 = [res1.x[2]-pden*res1.x[3],-res2.x[2]+pden*res2.x[3]] # position of interface - half width
    cutoffs2 = [res1.x[2]+pdil*res1.x[3],-res2.x[2]-pdil*res2.x[3]] # get far enough from interface for dilute phase calculation

    bool1 = np.logical_and(z<cutoffs1[0],z>cutoffs1[1])
    bool2 = np.logical_or(z>cutoffs2[0],z<cutoffs2[1])

    dilarray = np.apply_along_axis(lambda a: a[bool2].mean(), 1, hX) # concentration in range [bool2]
    denarray = np.apply_along_axis(lambda a: a[bool1].mean(), 1, hX)

    if (np.abs(cutoffs2[1]/cutoffs2[0]) > 2) or (np.abs(cutoffs2[1]/cutoffs2[0]) < 0.5): # ratio between right and left should be close to 1
        print('NOT CONVERGED',name,cutoffs1,cutoffs2)
        print(res1.x,res2.x)

    if plot:
        N = 10
        fig, ax = plt.subplots(1,2,figsize=(8,4))
        ax[0].plot(z,hmref,color='black')
        ax[0].plot(z,hmX,color='C0')

        for c1,c2 in zip(cutoffs1,cutoffs2):
            ax[0].axvline(c1,color='gray')
            ax[0].axvline(c2,color='black')
        ax[0].set(xlabel='z [nm]', ylabel='Concentration [mM]')
        ax[0].set(yscale='log')
        ax[0].set(title=name)
        xs = np.arange(len(hX))
        dilrunavg = np.convolve(dilarray, np.ones(N)/N, mode='same')
        ax[1].plot(xs,dilarray)
        # ax[1].plot(xs,dilrunavg)
        ax[1].set(xlabel='Timestep',ylabel='cdil [mM]')
        fig.tight_layout()

    cdil = hmX[bool2].mean() # average concentration
    cden = hmX[bool1].mean()

    block_dil = BlockAnalysis(dilarray)
    block_den = BlockAnalysis(denarray)
    block_dil.SEM()
    block_den.SEM()

    value.loc[name,f'{X}_dil'] = cdil # block_dil.av
    value.loc[name,f'{X}_den'] = cden #block_den.av

    error.loc[name,f'{X}_dil'] = block_dil.sem
    error.loc[name,f'{X}_den'] = block_den.sem

    return(value, error)

def calcProfile_simple(name,temp,seq,L,vv,tmin=1200,tmax=None,step=1,fbase='.',
    plot=False,pden=3.,pdil=6.,dGmin=-7):
    h = np.load(f'{fbase}/{name}/{temp}/{name}.npy')
    N = len(seq)
    conv = 100/6.022/N/L/L*1e3
    h = h[tmin:tmax:step]*conv # corresponds to (binned) concentration in mM
    print(f'len h: {len(h)}')
    lz = h.shape[1]+1
    edges = np.arange(-lz/2.,lz/2.,1)/10
    dz = (edges[1]-edges[0])/2.
    z = edges[:-1]+dz
    profile = lambda x,a,b,c,d : .5*(a+b)+.5*(b-a)*np.tanh((np.abs(x)-c)/d) # hyperbolic function, parameters correspond to csat etc.
    residuals = lambda params,*args : ( args[1] - profile(args[0], *params) )
    hm = np.mean(h,axis=0)
    z1 = z[z>0]
    h1 = hm[z>0]
    z2 = z[z<0]
    h2 = hm[z<0]
    p0=[1,1,1,1]
    res1 = least_squares(residuals, x0=p0, args=[z1, h1], bounds=([0]*4,[100]*4)) # fit to hyperbolic function
    res2 = least_squares(residuals, x0=p0, args=[z2, h2], bounds=([0]*4,[100]*4))

    cutoffs1 = [res1.x[2]-pden*res1.x[3],-res2.x[2]+pden*res2.x[3]] # position of interface - half width
    cutoffs2 = [res1.x[2]+pdil*res1.x[3],-res2.x[2]-pdil*res2.x[3]] # get far enough from interface for dilute phase calculation

    bool1 = np.logical_and(z<cutoffs1[0],z>cutoffs1[1]) # dense
    bool2 = np.logical_or(z>cutoffs2[0],z<cutoffs2[1]) # dilute

    denarray = np.apply_along_axis(lambda a: a[bool1].mean(), 1, h)
    dilarray = np.apply_along_axis(lambda a: a[bool2].mean(), 1, h) # concentration in range [bool2]

    # print(f'dilarray: {dilarray}')
    # print(f'denarray: {denarray}')

    if (np.abs(cutoffs2[1]/cutoffs2[0]) > 2) or (np.abs(cutoffs2[1]/cutoffs2[0]) < 0.5): # ratio between right and left should be close to 1
        print('NOT CONVERGED',name,cutoffs1,cutoffs2)
        print(res1.x,res2.x)

    if plot:
        os.system('mkdir -p figures')
        N = 10
        fig, ax = plt.subplots(1,2,figsize=(8,4))
        ax[0].plot(z,hm)
        for c1,c2 in zip(cutoffs1,cutoffs2):
            ax[0].axvline(c1,color='gray')
            ax[0].axvline(c2,color='black')

        ax[0].set(xlabel='z [nm]', ylabel='Concentration [mM]')
        ax[0].set(yscale='log')
        ax[0].set(title=name)
        xs = np.arange(len(h))
        dilrunavg = np.convolve(dilarray, np.ones(N)/N, mode='same')
        ax[1].plot(xs,dilarray)
        ax[1].set(xlabel='Timestep',ylabel='cdil [mM]')
        fig.tight_layout()
        fig.savefig(f'figures/{name}_{temp}.pdf')

    cutoffs1 = np.array(cutoffs1)
    cutoffs2 = np.array(cutoffs2)

    np.save(f'{fbase}/{name}/{temp}/cutoffs_dense.npy',cutoffs1)
    np.save(f'{fbase}/{name}/{temp}/cutoffs_dilute.npy',cutoffs2)

    cdil = hm[bool2].mean() # average concentration # CHANGED
    cden = hm[bool1].mean() # CHANGED

    print(f'dil: {cdil}')
    print(f'den: {cden}')

    # dGs = np.log(dilarray/denarray)

    block_dil = BlockAnalysis(dilarray)
    block_den = BlockAnalysis(denarray)
    block_dil.SEM()
    block_den.SEM()

    # cdil = block_dil.av
    # cden = block_den.av

    edil = block_dil.sem
    eden = block_den.sem

    # block_dG = BlockAnalysis(dGs)
    # block_dG.SEM()

    if np.isnan(cdil) or np.isnan(cden):
        print("Not converged, setting dG to 0")
        dG = 0.
        dG_error = 0.
    elif cdil == 0.:
        print(f'No dilute phase, setting dG to {dGmin:.1f}')
        dG = dGmin
        dG_error = 0.
    else:
        dG, dG_error = analysis.calc_dG(cdil,edil,cden,eden,ndraws=10000)
    if dG < dGmin:
        dG = dGmin
        dG_error = 0.
        print(f'dG extremely small, setting dG to {dGmin:.1f}')
    vv.loc[name,'nframes'] = len(h)

    vv.loc[name,'cdil'] = cdil
    vv.loc[name,'cden'] = cden

    vv.loc[name,'edil'] = edil
    vv.loc[name,'eden'] = eden

    vv.loc[name,'dG'] = dG
    vv.loc[name,'dG_error'] = dG_error

    return(vv)

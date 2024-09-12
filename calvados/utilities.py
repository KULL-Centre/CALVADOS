import pandas as pd
import MDAnalysis as mda
from Bio import SeqIO, SeqUtils
import yaml

def xconv(x,N=5):
    xf = np.convolve(x, np.ones(N)/N, mode='same')
    return xf

def autocorr(x,norm=True):
    y = x.copy()
    if norm:
        x = (x - np.mean(x)) / (np.std(x) * len(x))
        y = (y - np.mean(y)) / (np.std(y))
    c = np.correlate(x,y,mode='full')
    c = c[len(c)//2:]
    return c
import json
import os
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .inputmodels import InputPath


def xconv(x: NDArray[np.float64], N: int = 5) -> NDArray[np.float64]:
    """Smooth an array with a centered moving-average convolution."""
    xf = np.convolve(x, np.ones(N)/N, mode='same')
    return xf

def autocorr(x: NDArray[np.float64], norm: bool = True) -> NDArray[np.float64]:
    """Calculate the nonnegative-lag autocorrelation of a one-dimensional array."""
    y = x.copy()
    if norm:
        x = (x - np.mean(x)) / (np.std(x) * len(x))
        y = (y - np.mean(y)) / (np.std(y))
    c = np.correlate(x,y,mode='full')
    c = c[len(c)//2:]
    return c

def calc_runavg(xs: NDArray[np.float64], N: int = 10) -> NDArray[np.float64]:
    """Calculate a NaN-aware running average over ``N`` neighbors per side."""
    xs_ravg = []
    for idx, x in enumerate(range(len(xs))):
        x0 = max(0,idx-N)
        x1 = min(len(xs), idx+N+1)
        y = np.nanmean(xs[x0:x1])
        xs_ravg.append(y)
    return np.array(xs_ravg, dtype=np.float64)

def write_entry(
    uniprot: str,
    entry: Any,
    pdb_folder: InputPath,
    ) -> None:
    """Write an AlphaFold database entry to a JSON metadata file."""
    with open(f'{pdb_folder}/{uniprot}_info.json','w') as f:
        json.dump(entry,f)

def load_ebi(
    uniprot: str,
    pdb_folder: InputPath,
) -> None:
    """Download an AlphaFold structure, PAE matrix, and metadata from EBI."""
    os.system(f'mkdir -p {pdb_folder}')
    with os.popen(f'curl https://alphafold.ebi.ac.uk/api/prediction/{uniprot}') as f:
        entry = f.read()
    entry = json.loads(entry)[0]
    os.system(f'curl -L {entry["pdbUrl"]} -o {pdb_folder}/{uniprot}.pdb')
    os.system(f'curl -L {entry["paeDocUrl"]} -o {pdb_folder}/{uniprot}.json')
    write_entry(uniprot,entry,pdb_folder)

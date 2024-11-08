import numpy as np

def blocker(array, multi=1):
    dimension = len(array)
    rep = dimension/multi
    n_blocks_try = np.arange([2 if multi==1 else multi][0],dimension+1)
    n_blocks = []
    block_sizes = []

    for n in n_blocks_try:
        bs = dimension/n
        if (dimension % n == 0) & (rep % bs == 0):
            n_blocks.append(int(n))
            block_sizes.append(bs)

    return dimension, np.array(n_blocks), block_sizes

def check(array, multi=1):
    nt = len( blocker(array, multi=multi)[1] )
    if nt > 19:
        #print ("Possible blocks transformations: "+str(nt)+"\n no lenght correction needed\n")
        return array
    else:
        replen = int(len(array) / multi)
        for c in range(1,102):
            #print ("Removing "+str(c)+" at the bottom of each replica")
            chunks_array = np.array([])
            for n in range(1,multi+1):
                e = replen*n
                s = e - replen
                chunks_array = np.concatenate((chunks_array,array[s:e-c]))
            nt = len( blocker(chunks_array, multi=multi)[1] )
            #print ("Possible blocks transformations: "+str(nt)+"\n")
            if nt > 19:
                break
        return chunks_array

 
def blocking(array, multi=1):
    
    u = array.mean()
    N, n_blocks, block_sizes = blocker(array, multi=multi)
    
    errs = []
    errs_errs = []
    for b in range(len(block_sizes)):
        Nb = n_blocks[b]
        blocks_av = np.zeros(Nb)
        for n in range(1,Nb+1):
            end = int( block_sizes[b] * n )
            start = int( end - block_sizes[b] )
            blocks_av[n-1] = array[start:end].mean()

        err = np.sqrt( ((blocks_av - u)**2).sum() / (Nb*(Nb-1)) )
        errs.append(err)

        err_err = err/(np.sqrt(2*(Nb-1)))

        errs_errs.append(err_err)

    return np.flip( np.array([block_sizes, errs, errs_errs]).T , axis=0  )

def fblocking(cv, w, kbt, multi=1, interval=None):

    N, n_blocks, block_sizes = blocker(cv, multi=multi)
    u, bins = np.histogram(cv,weights=w,bins=50,range=(interval[0],interval[1]))
    zero_ndx = np.where(u==0)    
    u = np.delete(u, zero_ndx)
    bins = np.delete(bins, zero_ndx)
    u = u/N
    
    err = np.zeros(len(block_sizes))
    err_err = np.zeros(len(block_sizes))
    for b in range(len(block_sizes)):
        Nb = n_blocks[b]
        his = np.zeros(len(bins)-1)
        for n in range(Nb):
            start = int( n*block_sizes[b] )
            end = int( start + block_sizes[b] )
            hi = np.histogram(cv[start:end], weights=w[start:end], bins=bins)[0] / len(cv[start:end])
            his += (hi-u)**2
        e = np.sqrt( his / (Nb*(Nb-1)) )
        e = kbt*e/u
        err[b] += e.mean()
        err_err[b] += err[b] / np.sqrt( 2*(Nb-1) )
    
    return np.flip( np.array([block_sizes, err, err_err]).T , axis=0  )

def autocorrelation(x):
    n = len(x)
    variance = x.var()
    x = x-x.mean()
    r = np.correlate(x, x, mode = 'full')[-n:]
    result = r/(variance*(np.arange(n, 0, -1)))
    return result

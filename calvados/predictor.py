# import calvados as cal
# from calvados.build import calc_box, calc_nprot_slab
from .sequence import SeqFeatures

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numba as nb

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from scipy.stats import linregress, sem, spearmanr, brunnermunzel
from scipy.spatial.distance import mahalanobis
from scipy.optimize import curve_fit

from itertools import combinations
from copy import deepcopy

from tqdm import tqdm
from joblib import dump, load

from random import shuffle

from yaml import safe_load

import cProfile
import pstats

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

def bin_data(xs,ys,nbins,drange=None):
    """ bin data ys in xs bins, based on numpy.histogram_bin_edges """
    if drange == None:
        xmin, xmax = np.min(xs), np.max(xs)
    else:
        xmin, xmax = drange[0], drange[1]
    bins = np.linspace(xmin,xmax,nbins+1)
    y_binned = [[] for _ in range(nbins)]
    for x, y in zip(xs,ys):
        if x <= bins[0]:
            y_binned[0].append(y)
        elif x >= bins[-1]:
            y_binned[nbins-1].append(y)
        else:
            for idx in range(nbins):
                if x >= bins[idx] and x < bins[idx+1]:
                    y_binned[idx].append(y)
    return bins, y_binned

def name_to_index(df,name):
    return df.index[df['seq_name'] == name]

def predict_single(X,model):
    y = model.predict(X)
    return y

def predict_multimodels(X,models):
    ys = np.zeros(len(models))
    for idx, model in enumerate(models):
        ys[idx] = predict_single(X,model)
    return ys

def scan_around_single(X_orig,feats,models,limits=None,df=None,fcolor=plt.cm.Blues):
    for jdx, feat in enumerate(feats):
        X = X_orig.copy()
        axij = ax[jdx]
        replace_idx = feats.index(feat)
        
        if feat in limits:
            xs = np.linspace(limits[feat][0],limits[feat][1],100)
        else:
            xs = np.linspace(df[feat].min(),df[feat].max(),100)
        
        for x in xs:
            X[0,replace_idx] = x
            # print(X)
            ys = np.zeros(len(models))
        
            for idx, model in enumerate(models):
                ys[idx] = predict_single(model,X)
        
            # print(f'mean: {np.mean(ys)}')
            # print(f'std: {np.std(ys)}')
            markers, caps, bars = axij.errorbar(x,np.mean(ys),yerr=np.std(ys),marker='.',capsize=2,c=fcolor(0))
            [bar.set_alpha(0.5) for bar in bars]
            [cap.set_alpha(0.5) for cap in caps]
        axij.set_title(f'Scan {feats[replace_idx]}')
    
    ys = np.zeros(len(models))
    for idx, model in enumerate(models):
        ys[idx] = predict_single(model,X_orig)
    for jdx in range(len(feats)):
        ax[jdx].errorbar(X_orig[0,jdx],np.mean(ys),yerr=np.std(ys),marker='o',capsize=3,c='tab:red')
    return ys

def X_from_seq(seq,feats,residues=[],charge_termini=True,nu_file=None,ah_intgrl_map=None,lambda_map=None,
              seq_feats=None):
    X = []
    if seq_feats == None:
        seq_feats = SeqFeatures(seq,residues=residues,charge_termini=charge_termini,nu_file=nu_file,
                                       ah_intgrl_map=ah_intgrl_map,lambda_map=lambda_map)
    for feat in feats:
        X.append(getattr(seq_feats,feat))
    X = np.array(X)
    X = np.reshape(X,(1,-1))
    return X

def makeXy(df,feats,target=None):
    """ Make feature (X) -- target (y) pairs from dataframe """
    X, y, X_keys = [], [], []
    
    for key, val in df.iterrows():
        features = []

        for feat in feats: # feats is a list of string
            features.append(val[feat]) # features is a list of values

        X.append(features)
        X_keys.append(key)

        if target is not None:
            target_sim = val[target]
            y.append(target_sim)

    X = np.array(X)
    if target is not None:
        y = np.array(y)
        return X, y, X_keys
    else:
        return X, X_keys

class AttrSetter:
    def __init__(self,**kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

class Model:
    def __init__(self,**kwargs):
        self.mltype = kwargs.get('mltype','svr')
        self.layers = kwargs.get('layers',(5,5))
        self.alpha = kwargs.get('alpha',10)
        self.C = kwargs.get('C',10)
        self.epsilon = kwargs.get('epsilon',1e-2)
        self.ptrain = kwargs.get('ptrain',0.8)
        self.ncrossval = kwargs.get('ncrossval',50)

    @staticmethod
    def split_data(X,y,X_keys,ptrain):
        """ Split data into train and test set and return corresponding indices """
        nsamp = len(X)
        if nsamp != len(y):
            raise ValueError("X and y size is not equal!")
    
        random_idx = np.random.choice(nsamp, size=nsamp, replace=False)
        ntrain = int(nsamp * ptrain)
        train_idx = random_idx[:ntrain]
        test_idx = random_idx[ntrain:]
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        X_train_keys = [X_keys[idx] for idx in train_idx]
        X_test_keys = [X_keys[idx] for idx in test_idx]
    
        return X_train, X_test, y_train, y_test, X_train_keys, X_test_keys
    
    @staticmethod
    def calc_statistics(y, ypred, verbose=True):
        # Pearson
        fit = linregress(y, ypred)
        rp = fit.rvalue

        # Spearmanx
        rs = spearmanr(y, ypred).statistic

        # Root mean squared deviation
        rmsd = np.sqrt(np.mean((y - ypred)**2))

        if verbose:
            print(f'Pearson: {rp:.3f}, Spearman: {rs:.3f}, RMSD: {rmsd:.3f}')
        return rp, rs, rmsd

    @staticmethod
    def calc_statistics_multimodel(y, ypred, verbose=True):
        nmodels = len(ypred)
        
        rp = np.zeros((nmodels))
        rs = np.zeros((nmodels))
        rmsd = np.zeros((nmodels))
        
        for idx, yp in enumerate(ypred):
            rp[idx], rs[idx], rmsd[idx] = Model.calc_statistics(y, yp, verbose=verbose)
        return rp, rs, rmsd

    def predict(self,X):
        ypred = np.zeros((self.ncrossval, len(X)))
        for idx, crossval in enumerate(self.crossvals):
            ypred[idx] = crossval.mlmodel.predict(X)
        return ypred

    def train(self,X,y,X_keys,**kwargs):
        self.models = []
        self.crossvals = []
        verbose = kwargs.get('verbose',True)

        for idx in range(self.ncrossval):
            X_train, X_test, y_train, y_test, X_train_keys, X_test_keys = self.split_data(X,y,X_keys,self.ptrain)

            if self.mltype == 'svr':
                mlmodel = make_pipeline(StandardScaler(), SVR(C=self.C, epsilon=self.epsilon))
            elif self.mltype == 'mlp':
                mlmodel = make_pipeline(
                    StandardScaler(),
                    MLPRegressor(
                        hidden_layer_sizes=self.layers,activation='tanh',
                        solver='lbfgs',max_iter=10000,alpha=self.alpha),
                )
            mlmodel.fit(X_train, y_train)

            ypred_train = mlmodel.predict(X_train)
            ypred_test = mlmodel.predict(X_test)

            rp, rs, rmsd = self.calc_statistics(y_test, ypred_test, verbose=verbose)

            self.crossvals.append(AttrSetter(
                X_train = X_train,
                X_test = X_test,
                y_train = y_train,
                y_test = y_test,
                X_train_keys = X_train_keys,
                X_test_keys = X_test_keys,
                mlmodel = mlmodel,
                ypred_train = ypred_train,
                ypred_test = ypred_test,
                rp = rp,
                rs = rs,
                rmsd = rmsd
            ))
        self.rp_mean = np.mean([cval.rp for cval in self.crossvals])
        self.rs_mean = np.mean([cval.rs for cval in self.crossvals])
        self.rmsd_mean = np.mean([cval.rmsd for cval in self.crossvals])

def add_seq(df,records):
    for key, val in df.iterrows():
        add = False
        if 'fasta' not in df.keys():
            add = True
        elif not isinstance(val['fasta'], str):
            add = True
        if add:
            if key in records:
                df.loc[key,'fasta'] = str(records[key].seq)
            else:
                print(f'Could not find {key} in records')
    return df

def add_seq_from_df_full(df,df_full):
    for name, val in df.iterrows():
        if not isinstance(val['fasta'], str):
            key = name_to_index(df_full,name)[0]
            print(key)
            df.loc[name,'fasta'] = df_full.loc[key,'fasta']
    return df

def calc_cdil_mgml(N,mw,val):
    box = calc_box(N)
    nprot = calc_nprot_slab(N,box)

    if val['dG'] == 0.:
        cdil = nprot / (box[0] * box[1] * box[2]) / 6.022e23 * 1e24 * 1e3 # mM, bulk
    else:
        cdil = val['cdil']

    if cdil < np.exp(-7.5):
        cdil = np.exp(-7.5)
        logcdil = -7.5
    else:
        logcdil = np.log(cdil)

    cdil_mgml = cdil * mw  / 1e3

    if cdil_mgml < np.exp(-4):
        cdil_mgml = np.exp(-4)
        logcdil_mgml = -4.
        log10cdil_mgml = -1.737 # np.log10(np.exp(-4))
    else:
        logcdil_mgml = np.log(cdil_mgml)
        log10cdil_mgml = np.log10(cdil_mgml)
    
    return cdil, logcdil, cdil_mgml, logcdil_mgml, log10cdil_mgml

def convert_cdil(df):
    for key, val in df.iterrows():
        seq = val['fasta']
        N = len(seq)
        # calc mw
        mw = calc_mw(seq)
        df.loc[key,'mw'] = mw
        # convert cdil
        cdil, logcdil, cdil_mgml, logcdil_mgml, log10cdil_mgml = calc_cdil_mgml(N,mw,val)
        df.loc[key,'cdil'] = cdil
        df.loc[key,'logcdil'] = logcdil
        df.loc[key,'cdil_mgml'] = cdil_mgml
        df.loc[key,'logcdil_mgml'] = logcdil_mgml
        df.loc[key,'log10cdil_mgml'] = log10cdil_mgml
    return df

def add_features(df,feats,charge_termini=True,residues=None,nu_file=None,verbose=False,
ah_intgrl_map=None, lambda_map=None):
    for key, val in tqdm(df.iterrows(),total=len(df)):
        seq = val['fasta']
        if verbose:
            print(key, seq)
        N = len(seq)
        seq_feats = SeqFeatures(seq,charge_termini=charge_termini,
                                            residues=residues,nu_file=nu_file,
                                            ah_intgrl_map=ah_intgrl_map,lambda_map=lambda_map)
        for feat in feats:
            df.loc[key,feat] = getattr(seq_feats,feat)
    return df
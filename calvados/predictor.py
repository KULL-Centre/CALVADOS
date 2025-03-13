import numpy as np
import calvados as cal
from .sequence import SeqFeatures
import pandas as pd
from tqdm import tqdm
from joblib import dump, load

from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor

from scipy.stats import linregress, sem, spearmanr, brunnermunzel

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

def X_from_seq(seq,feats,residues=[],charge_termini=True,nu_file=None,ah_intgrl_map=None,lambda_map=None,
              seq_feats=None,flexhis=False,pH=None):
    X = []
    if seq_feats == None:
        seq_feats = SeqFeatures(seq,residues=residues,charge_termini=charge_termini,nu_file=nu_file,
                                       ah_intgrl_map=ah_intgrl_map,lambda_map=lambda_map,
                               flexhis=flexhis,pH=pH)
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
            # print(len(X_train),len(X_test),len(y_train),len(y_test))
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

def add_features(df,feats,charge_termini=True,residues=None,nu_file=None,verbose=False,
ah_intgrl_map=None, lambda_map=None, check_flexhis=False):
    for key, val in tqdm(df.iterrows(),total=len(df)):
        seq = val['fasta']
        if verbose:
            print(key, seq)
        N = len(seq)
        if check_flexhis:
            flexhis, pH = val['flexhis'], val['pH']
        else:
            flexhis, pH = False, None
        seq_feats = SeqFeatures(seq,charge_termini=charge_termini,
                                            residues=residues,nu_file=nu_file,
                                            ah_intgrl_map=ah_intgrl_map,lambda_map=lambda_map,
                                           flexhis=flexhis, pH=pH)
        for feat in feats:
            df.loc[key,feat] = getattr(seq_feats,feat)
    return df

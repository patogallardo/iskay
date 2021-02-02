'''Data handlers for opening and querying the ApPhoto_ApPhoto dataset.'''
import pandas as pd
import os
import numpy as np
import progressbar


FOLDER = './ApPhoto_ApPhoto'  # folder where datasets live
FNAMES = ['ApPhoto_090.hdf', 'ApPhoto_150.hdf']


def load_datasets(folder=FOLDER, fnames=FNAMES, query=None):
    df1 = pd.read_hdf(os.path.join(folder, fnames[0]), 'df')
    df2 = pd.read_hdf(os.path.join(folder, fnames[1]), 'df')
    if query is not None:
        df1.query(query, inplace=True)
        df2.query(query, inplace=True)
    return df1, df2


def resample_datasets(df1, df2, params):
    assert len(df1) == len(df2)
    df1_replicants = []
    df2_replicants = []
    print('Generating %s replicant samples:' % params.JK_RESAMPLING_METHOD)
    if 'jk' == params.JK_RESAMPLING_METHOD:
        all_indx = np.arange(len(df1))
        np.random.shuffle(all_indx)
        indx_to_drop = np.array_split(all_indx, params.JK_NGROUPS)
        for j in progressbar.progressbar(range(params.JK_NGROUPS)):
            todrop = indx_to_drop[j]
            tojk1 = df1.drop(df1.index[todrop], inplace=False)
            tojk2 = df2.drop(df2.index[todrop], inplace=False)

            df1_replicants.append(tojk1)
            df2_replicants.append(tojk2)
    elif 'bootstrap' == params.JK_RESAMPLING_METHOD:
        for j in progressbar.progressbar(range(params.JK_NGROUPS)):
            indxs = np.random.randint(low=0, high=len(df1),
                                      size=len(df1))
            df1_replicants.append(df1.iloc[indxs])
            df2_replicants.append(df2.iloc[indxs])
    else:
        assert(False)
    return df1_replicants, df2_replicants


class cross_dataset:
    def __init__(self, params, folder=FOLDER, fnames=FNAMES,
                 multiplyDataset1Times=1.0):
        query = params.CAT_QUERY
        df1, df2 = load_datasets(query=query)
        df1.dT = df1.dT.values * multiplyDataset1Times
        self.df1, self.df2 = df1, df2
        self.query = query
        self.names = [fname.split('_')[1].split('.')[0]
                      for fname in fnames]
#        replicant1, replicant2 = resample_datasets(df1, df2, params)
#        self.replicant1 = replicant1
#        self.replicant2 = replicant2

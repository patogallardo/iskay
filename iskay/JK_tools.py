'''JK tools for statistics, covariance matrices, dropping data for JK
runs, etc.

Written by P. Gallardo.'''

import numpy as np
import pandas as pd


def indicesToDrop(df, Ngroups, randomize=True):
    '''Receives a pandas dataframe and returns the indices that need to
    be dropped in every JackKnife iteration to have a Ngroups JackKnife.

    It will return a list of numpy arrays. There will be Ngroups numpy arrays.
    '''
    indices = df.index.values
    if randomize:
        np.random.seed(4)  #to make it predictable
        np.random.shuffle(indices)
    return np.array_split(indices, Ngroups)


def getErrorbars(jk_results, params):
    '''Computes jk errorbars from jk realizations '''
    assert params.JK_NGROUPS == jk_results.shape[0]
    sigma = np.std(jk_results, axis=0)
    sigma_sq = sigma**2
    errorbars_sq = (params.JK_NGROUPS-1) * sigma_sq
    errorbars = np.sqrt(errorbars_sq)
    return errorbars


#covariance matrix
def getBinNames(rsep):
    '''For a given rsep, makes a list of bin names for use with pandas
       dataframe'''
    names = ["0 - %i" % rsep[0]]
    names += ["%i - %i" % (rsep[j], rsep[j+1]) for j in range(len(rsep)-1)]
    return names


def getCovMatrix(bin_names, pests):
    '''Uses pandas to get the covariance matrix of pests.
    bin_names are the names of the columns of pests.
    This assumes the pests vector was generated in a JK run,
    and uses the number of realizations to inflate the variances'''
    N = pests.shape[0]
    df = pd.DataFrame(pests, columns=bin_names)
    cov = df.cov() * (N-1) / N * (N-1)  # first term cancels N-1 from numpy
    return cov  # covariance. Second term replaces it by N, third term
#               #  compensates JK covariance.


def getCorrMatrix(bin_names, pests):
    '''Compute the correlation matrix using pandas.
    No inflation pre-factor is needed here as they cancel out.'''
    df = pd.DataFrame(pests, columns=bin_names)
    corr = df.corr()
    return corr
#end covariance matrix functions

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
    indices = df.index.values.copy()  # if you dont do .copy
    if randomize:  # the randomizer will give same result
        np.random.shuffle(indices)  # for diffferent runs
    groups = np.array_split(indices, Ngroups)  # seems to be
    return groups  # a weird bug in numpy/pandas, fixed in
    # later versions


def getErrorbars(jk_results, params):
    '''Computes jk errorbars from jk realizations
    Jun 2020: added support for bootstrap realizations'''
#    assert params.JK_NGROUPS == jk_results.shape[0] and not IS_TILEDJK
    sigma = np.std(jk_results, axis=0)
    sigma_sq = sigma**2
    errorbars_sq = (params.JK_NGROUPS-1) * sigma_sq
    JK_errorbars = np.sqrt(errorbars_sq)

    if 'jk' == params.JK_RESAMPLING_METHOD.lower():
        return JK_errorbars
    elif 'bootstrap' in params.JK_RESAMPLING_METHOD.lower():
        return sigma
    elif 'bs_pairwise' in params.JK_RESAMPLING_METHOD.lower():
        return sigma
    elif 'bs_dt' in params.JK_RESAMPLING_METHOD.lower():
        return sigma
    elif 'tiled_jk' in params.JK_RESAMPLING_METHOD.lower():
        NGroups = jk_results.shape[0]
        errorbars_sq = NGroups * sigma_sq
        sigma = np.sqrt(errorbars_sq)
        return sigma
    else:
        assert False  # resampling error.


#covariance matrix
# supreceded by getBinNamesFromBinEdges
# ############### this function was dropped in 4/27/2020 to
# ###############handle uneven bins
def getBinNames(rsep):
    '''For a given rsep, makes a list of bin names for use with pandas
       dataframe'''
    names = ["0 - %i" % rsep[0]]
    names += ["%i - %i" % (rsep[j], rsep[j+1]) for j in range(len(rsep)-1)]
    return names
###################


def getBinNamesFromParams(params):
    if params.UNEVEN_BINS:
        names = getBinNamesFromBinEdges(params.BIN_EDGES)
    else:
        names = getBinNamesFromNbinsBinsize(params.N_BINS,
                params.BIN_SIZE_MPC) # noqa
    return names


def getBinNamesFromNbinsBinsize(nbins, binsize):
    bin_edges = np.arange(0, (nbins+1)*binsize)
    names = getBinNamesFromBinEdges(bin_edges)
    return names


def getBinNamesFromBinEdges(bin_edges):
    '''For givne bin edges, make labels for these bins.'''
    names = ["%i - %i" % (bin_edges[j], bin_edges[j+1])
             for j in range(len(bin_edges)-1)]
    return names


def getCovMatrix(bin_names, pests, params):
    '''Uses pandas to get the covariance matrix of pests.
    bin_names are the names of the columns of pests.
    This assumes the pests vector was generated in a JK run,
    and uses the number of realizations to inflate the variances'''
    N = pests.shape[0]
    df = pd.DataFrame(pests, columns=bin_names)
    if 'jk' == params.JK_RESAMPLING_METHOD.lower():
        cov = df.cov() * (N-1.) / N * (N-1.)  # first term cancels N-1 from np
    if 'tiled_jk' == params.JK_RESAMPLING_METHOD.lower():
        N_groups = pests.shape[0]
        cov = df.cov() * (N_groups-1.)/N_groups * (N_groups-1.)
    elif 'bootstrap' in params.JK_RESAMPLING_METHOD.lower():
        cov = df.cov()
    elif 'bs_pairwise' in params.JK_RESAMPLING_METHOD.lower():
        cov = df.cov()
    elif 'bs_dt' in params.JK_RESAMPLING_METHOD.lower():
        cov = df.cov()
    return cov  # covariance. Second term replaces it by N, third term
#               #  compensates JK covariance.


def getJointCovMatrix(bin_names, pests1, pests2, params):
    assert pests1.shape[0] == pests2.shape[0]
    N = pests1.shape[0]

    df1 = pd.DataFrame(pests1, columns=bin_names)
    df2 = pd.DataFrame(pests2, columns=bin_names)
    df_joint = pd.concat([df1, df2], axis=1)

    if 'jk' == params.JK_RESAMPLING_METHOD.lower():
        cov = df_joint.cov() * (N-1.) / N * (N-1.)
    elif 'bootstrap' in params.JK_RESAMPLING_METHOD.lower():
        cov = df_joint.cov()
    return cov


def getCorrMatrix(bin_names, pests):
    '''Compute the correlation matrix using pandas.
    No inflation pre-factor is needed here as they cancel out.'''
    df = pd.DataFrame(pests, columns=bin_names)
    corr = df.corr()
    return corr
#end covariance matrix functions

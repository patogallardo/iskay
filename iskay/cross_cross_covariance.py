import numpy as np


def cross_cov(x, y, jackknife):
    '''Computes the cross covariance between x, y.
    x,y: matrices of nrealization, nbin shape
    jackknife = True if covariance inflation is applied.
    False for regular covarinace.'''
    if jackknife is False:
        prefactor = 1.0
    else:
        prefactor = x.shape[0] - 1.0
    d = x.shape[0] - 1.0

    mux = np.mean(x, axis=0)
    muy = np.mean(y, axis=0)

    x_mean_corrected = x - mux
    y_mean_corrected = y - muy

    C = prefactor/d * np.dot(x_mean_corrected.T, y_mean_corrected)

    return C

import numpy as np


def weighted_median(x, w):
    '''Computes the weighted median according to
    https://en.wikipedia.org/wiki/Weighted_median
    '''
    argsort = np.argsort(x)
    sorted_x = x[argsort]
    sorted_w = w[argsort]

    cs = np.cumsum(sorted_w)/np.sum(sorted_w)
    sel = cs < 0.5

    srtd_median_indx = np.where(sel)[0][-1]+1
    return sorted_x[srtd_median_indx]


def compute_weighted_median_or_mean(x, w=None,
                                    method='median'):
    if w is None and method == 'median':
        return np.median(x)
    elif w is not None and method == 'median':
        assert len(x) == len(w)
        return weighted_median(x, w)
    elif w is None and method == 'mean':
        return np.mean(x)
    elif w is not None and method == 'mean':
        assert len(x) == len(w)
        return np.average(x, weights=w)


def square_window(x):
    '''returns 0 if x<-1 and 1 if x>1'''
    return np.heaviside(x+1, 1) - np.heaviside(x-1, 0)


def make_weights(z, center_z, sigma_z, method='gaussian'):
    '''Receives list of redshifts, center redshift, sigma_z
    (window semiwidth, and select method: gaussian or
    square)'''
    argument = (z-center_z)/sigma_z
    if method == 'gaussian':
        return np.exp(-argument**2)
    elif method == 'square':
        return square_window(argument)

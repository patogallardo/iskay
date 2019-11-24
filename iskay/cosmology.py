'''Cosmological calculations to get Dc go here.
'''
import numba
import math as mt
import itertools
import numpy as np
from scipy.integrate import quad
from scipy import constants

Hub = 67.31
mat = 0.315
lam = 1.-mat
clight = constants.c/1000.  #km/s


@numba.jit(nopython=True)
def InvEz(z, H0, om, ol):
    '''Cosmological 1/E(z) function to integrate.'''
    return 1.0/(mt.sqrt(om*(1.0+z)**3 + ol)*H0)


def Dc(z):
    '''Loop over z and integrate InvEz'''
    n = len(z)
    zeros = itertools.repeat(0.0, n)
    f_toIntegrate = itertools.repeat(InvEz, n)
    args = itertools.repeat((Hub, mat, lam), n)
    res = map(quad, f_toIntegrate, zeros, z, args)
    res = np.array(res)
    return res[:, 0]*clight

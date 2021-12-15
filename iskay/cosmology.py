'''Cosmological calculations to get Dc go here.
'''
import numba
import math as mt
import itertools
import numpy as np
from scipy.integrate import quad
from scipy import constants
import pandas as pd
from scipy.interpolate import interp1d


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


def mk_varying_ap_photo_interpolator():
    fname = ('/home/pag227/code/iskay/misc/'
             'varyingApertureApPhotoData/Da_vs_z.dat')
    df_Da = pd.read_csv(fname, delim_whitespace=True)[['z', 'Da']]
    f_interp = interp1d(df_Da.z.values, df_Da.Da.values,
                        fill_value='extrapolate')
    return f_interp


def ap_photo_of_z(f_interp, z, ap_0=2.1, z_0=0.5):
    ''' Returns value of the aperture photomery, needs the interpolator
    made in mk_varying_ap_photo_interpolator and the redshift.
    ap_0 and z_0 are fiducial values, in our case we use 2.1 and 0.5
    '''
    f_interp_0p5 = f_interp(z_0)
    aperture = f_interp_0p5/f_interp(z) * ap_0
    return aperture


def r_disk_of_z(z):
    f_interp = mk_varying_ap_photo_interpolator()
    r_disk = ap_photo_of_z(f_interp, z)
    return r_disk

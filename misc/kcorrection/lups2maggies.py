''' Converts luptitudes to maggies and stores in folder output
    Written by P. Gallardo
'''
import numpy as np
import pandas as pd
import sys

assert len(sys.argv) == 2  # usage: lups2maggies.py /path/to/cat.csv
fname = sys.argv[1]

print("Converting maggies from catalog \n%s" % fname)

df = pd.read_csv(fname)

z = df['z'].values

mod_u = df['cModelMag_u'].values
mod_g = df['cModelMag_g'].values
mod_r = df['cModelMag_r'].values
mod_i = df['cModelMag_i'].values
mod_z = df['cModelMag_z'].values

ext_u = df['extinction_u'].values
ext_g = df['extinction_g'].values
ext_r = df['extinction_r'].values
ext_i = df['extinction_i'].values
ext_z = df['extinction_z'].values

err_u = df['cModelMagErr_u'].values
err_g = df['cModelMagErr_g'].values
err_r = df['cModelMagErr_r'].values
err_i = df['cModelMagErr_i'].values
err_z = df['cModelMagErr_z'].values


dered_u = mod_u - ext_u
dered_g = mod_g - ext_g
dered_r = mod_r - ext_r
dered_i = mod_i - ext_i
dered_z = mod_z - ext_z

b = np.array([1.4, 0.9, 1.2, 1.8, 7.4]) * 1e-10
flux_u = 2.*b[0] * np.sinh(-np.log(10.)/2.5*dered_u-np.log(b[0]))
flux_g = 2.*b[1] * np.sinh(-np.log(10.)/2.5*dered_g-np.log(b[1]))
flux_r = 2.*b[2] * np.sinh(-np.log(10.)/2.5*dered_r-np.log(b[2]))
flux_i = 2.*b[3] * np.sinh(-np.log(10.)/2.5*dered_i-np.log(b[3]))
flux_z = 2.*b[4] * np.sinh(-np.log(10.)/2.5*dered_z-np.log(b[4]))


ivar_u = 2.*b[0]*np.cosh(-np.log(10.)/2.5*dered_u-np.log(b[0]))*(-np.log(10)/2.5)*err_u # noqa
ivar_g = 2.*b[1]*np.cosh(-np.log(10.)/2.5*dered_g-np.log(b[1]))*(-np.log(10)/2.5)*err_g # noqa
ivar_r = 2.*b[2]*np.cosh(-np.log(10.)/2.5*dered_r-np.log(b[2]))*(-np.log(10)/2.5)*err_r # noqa
ivar_i = 2.*b[3]*np.cosh(-np.log(10.)/2.5*dered_i-np.log(b[3]))*(-np.log(10)/2.5)*err_i # noqa
ivar_z = 2.*b[4]*np.cosh(-np.log(10.)/2.5*dered_z-np.log(b[4]))*(-np.log(10)/2.5)*err_z # noqa

ivar_u = 1./ivar_u**2.
ivar_g = 1./ivar_g**2.
ivar_r = 1./ivar_r**2.
ivar_i = 1./ivar_i**2.
ivar_z = 1./ivar_z**2.

to_exp = np.transpose([z, flux_u, flux_g, flux_r, flux_i, flux_z,
                       ivar_u, ivar_g, ivar_r, ivar_i, ivar_z])
np.savetxt('./output/maggies.txt',
           to_exp)

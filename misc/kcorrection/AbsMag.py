import numpy as np
from scipy.integrate import quad
import math
import pandas as pd
import sys

Hub = 70.
mat = 0.274
lam = 1.-mat
clight = 299792.458  # Km/s


def InvEz(z, H0, om, ol):
    '''Cosmological generic 1/E(z) function, useful for integrating'''
    inve = 1./(np.sqrt(om*(1.+z)**3. + ol)*H0)
    return inve


fname = sys.argv[1]
print("Using catalog: %s" % fname)

rec = np.loadtxt('./output/X1.dat')
rec0_s01 = np.loadtxt('./output/X2.dat')
redc, uc, gc, rc, ic, zc = (rec[:, 0], rec[:, 1], rec[:, 2], rec[:, 3],
                            rec[:, 4], rec[:, 5])
redc0s01, uc0s01, gc0s01, rc0s01, ic0s01, zc0s01 = (rec0_s01[:, 0],
         rec0_s01[:, 1], rec0_s01[:, 2], rec0_s01[:, 3],  # noqa
         rec0_s01[:, 4], rec0_s01[:, 5])  # noqa


df1 = pd.read_csv(fname)

df = df1[['ra', 'dec', 'z',
          'cModelMag_u', 'cModelMag_g', 'cModelMag_r', 'cModelMag_i',
          'cModelMag_z',
          'cModelMagErr_u', 'cModelMagErr_g', 'cModelMagErr_r',
          'cModelMagErr_i', 'cModelMagErr_z',
          'extinction_u', 'extinction_g', 'extinction_r', 'extinction_i',
          'extinction_z', 'bestObjID']]

ra = df['ra'].values
dec = df['dec'].values
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

# comoving radial distances
DC = np.empty(len(z))
for i in range(len(z)):
    DC[i] = quad(InvEz, 0, z[i], args=(Hub, mat, lam))[0]*clight
# luminosity distance
DL = DC * (1. + z)

dMod_my = 5.*np.log10(DL*1e6/10.)

rat = rc/rc0s01  # see kcorrect manual
# calculate kcorrection
kkc = -2.5*np.log10(rat)

# redshift evolution (see Tegmark)
qz = 1.6*(z-0.1)
#aM = petro11 - dMod11 - kkc+ qz #+ 5.*np.log10(Hub/100.)
aMdr = dered_r - dMod_my - kkc + qz

# Abs Mag of Sun
Msun = 4.76
# get luminosities
lumsdr = 10.**((Msun-aMdr)/2.5)

sel = np.array(map(math.isnan, lumsdr))
sel = np.logical_not(sel)
lumsdr = lumsdr[sel]

ra = ra[sel]
dec = dec[sel]
z = z[sel]

dered_r = dered_r[sel]
ext_r = ext_r[sel]
aMdr = aMdr[sel]
kkc = kkc[sel]

#now append luminosity column to the original catalog
df1 = df1.loc[sel]
df1['lum'] = lumsdr

fname_out = fname.split('.csv')[0].split('/')[-1] + "_kcorrected.csv"
fname_out = './output/%s' % fname_out

df1.to_csv(fname_out)

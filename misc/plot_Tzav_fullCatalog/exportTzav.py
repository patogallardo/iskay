''' reads a preprocessed catalog and computes tzav for it.

Writes a file with the numpy array

Written by: P. Gallardo.
'''
import numpy as np  # noqa
from iskay import catalogTools # noqa
from iskay import pairwiser # noqa

#look in parent directory
df = catalogTools.preProcessedCat(directory='../ApPhotoResults').df

sigma_z = 0.01
dT = df.dT.values
z = df.z.values
tzav = pairwiser.get_tzav(dT, z, sigma_z)

np.savez('tzav_allCat', tzav)

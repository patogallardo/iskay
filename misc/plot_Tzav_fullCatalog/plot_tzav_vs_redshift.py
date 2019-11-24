import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
from iskay import catalogTools

df = catalogTools.preProcessedCat(directory='../ApPhotoResults').df
dT = df.dT.values
z = df.z.values
tzav = np.load('tzav_allCat.npz')['arr_0']

f = interpolate.interp1d(z, tzav)
z_int = np.linspace(z.min(), z.max(), 2000)
tzav_0 = f(z_int)

plt.plot(z_int, tzav_0, label='$T_{zav}$ full cat')

randomSamples = np.random.randint(0, len(dT), size=1000)
plt.scatter(z[randomSamples], dT[randomSamples], label='dTs (cat)')

plt.legend()
plt.xlabel('z')
plt.ylabel('$T_{zav}$')
plt.savefig('z_vs_tzav_completeCat_interpolated.pdf')
plt.close()

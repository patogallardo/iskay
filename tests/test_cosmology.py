from iskay import cosmology
import numpy as np
from itertools import repeat
import mpmath as mp

Hub = cosmology.Hub
mat = cosmology.mat
lam = cosmology.lam
clight = cosmology.clight


def test_InvEz():
    N = 100
    zs = np.random.uniform(low=0.01, high=3, size=N)
    InvEz = 1.0/(np.sqrt(mat*(1.0 + zs)**3 + lam) * Hub)
    cosmoInvEz = map(cosmology.InvEz, zs, repeat(Hub, N), repeat(mat, N),
                     repeat(lam, N))
    cosmoInvEz = np.array(cosmoInvEz)
    diff_sq = (cosmoInvEz - InvEz)**2
    chisq = np.sum(diff_sq)
    assert chisq < 1e-16


def test_Dc():
    def f(z):
        return 1/(mp.sqrt(mat*(1 + z)**3 + lam) * Hub)
    N = 100
    zs = np.random.uniform(low=0.1, high=3, size=N)
    intervals = [[0, z] for z in zs]
    integrals = map(mp.quad, repeat(f, N), intervals)
    numericIntegrals = clight * np.array([float(value) for value in integrals])
    Dc_iskay = cosmology.Dc(zs)
    chisq = ((Dc_iskay - numericIntegrals)**2).sum()
    assert chisq < 1e-19

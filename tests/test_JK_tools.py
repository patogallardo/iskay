import pandas as pd
import numpy as np
from iskay import JK_tools


def test_indicesToDrop():
    N = 20000
    a = np.random.random(N)
    df = pd.DataFrame({'a': a})
    groups = JK_tools.indicesToDrop(df, 4)
    assert len(groups) == 4
    assert len(groups[0]) == 5000


def test_getErrorbars():
    class p:
        def __init__(self, ngroups):
            self.JK_NGROUPS = ngroups
    howManyJKiterations = 50000
    fakePars = p(howManyJKiterations)
    res = np.random.normal(size=[howManyJKiterations, 20])

    errorbars = JK_tools.getErrorbars(res, fakePars)

    std_res = np.std(res, axis=0) * np.sqrt(howManyJKiterations - 1)
    diff_sq = (std_res - errorbars)**2
    assert diff_sq.sum() < 1e-10


def test_getBinNames():
    rsep = np.array([5, 10, 15, 20])
    names = JK_tools.getBinNames(rsep)
    assert names == ['0 - 5', '5 - 10', '10 - 15', '15 - 20']


def test_getCovMatrix():
    bin_names = ['0 - 5', '5 - 10', '10 - 15', '15 - 20']
    pests = np.random.random(size=[50, 4])
    N = 50
    cov = JK_tools.getCovMatrix(bin_names, pests)
    cov = cov.values
    cov_numpy = np.cov(pests.T) * (N-1)/N*(N-1)
    chi_sq = ((cov_numpy - cov)**2).flatten().sum()
    assert chi_sq < 1e-10


def test_getCorrMatrix():
    bin_names = ['0 - 5', '5 - 10', '10 - 15', '15 - 20']
    pests = np.random.random(size=[50, 4])
    corr = JK_tools.getCorrMatrix(bin_names, pests).values
    corr_numpy = np.corrcoef(pests.T)
    chi_sq = ((corr-corr_numpy)**2).flatten().sum()
    assert chi_sq < 1e-10

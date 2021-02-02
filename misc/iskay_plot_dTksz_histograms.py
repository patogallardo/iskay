from iskay import paramTools
from iskay import catalogTools
import matplotlib.pyplot as plt
from iskay import pairwiser
import numpy as np
import scipy.stats

fnames = ["params_disjoint_bin_lum_gt_04p3_and_06p1_jk.ini",
          "params_disjoint_bin_lum_gt_06p1_and_07p9_jk.ini",
          "params_lum_gt_07p9_jk.ini"]

for j in range(len(fnames)):
    fname = fnames[j]
    p = paramTools.params(fname)
    df = catalogTools.preProcessedCat(howMany=None, query=p.CAT_QUERY).df

    dT = df.dT.values
    z = df.z.values
    tzav = pairwiser.get_tzav_fast(dT, z, p.SIGMA_Z)
    dT_ksz = dT-tzav

    mean, std = np.mean(dT_ksz), np.std(dT_ksz)

    plt.figure(figsize=[8, 4.5])
    plt.hist(dT_ksz, normed=True, histtype='step', color='black', lw=2,
             bins=200)

    x = np.linspace(dT_ksz.min(), dT_ksz.max(), 1000)
    plt.plot(x, scipy.stats.norm.pdf(x, mean, std), color='C1',
             label="mu: %1.2e sigma=%1.2e" % (mean, std))
    plt.legend(fontsize=8)
    plt.xlabel('dT_ksz')
    plt.ylabel('pdf')
    plt.title(p.NAME)

    plt.savefig('results_jk/histograms_%s.png' % p.NAME)
    plt.close()

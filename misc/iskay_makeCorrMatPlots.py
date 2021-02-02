#!/nfs/user/pag227/miniconda/bin/python
import matplotlib
matplotlib.use('agg')

import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa
from iskay import JK  # noqa
import glob  # noqa
import os  # noqa
import numpy as np# noqa

font = {'size': 16}
matplotlib.rc('font', **font)

fnames = glob.glob('./results_*/*.pck')
assert len(fnames) > 0
show = True
showTitle = False


def getPath(fname):
    return '/'.join(fname.split('/')[:-1])


limitBins = 19  # show only these bins
for fname in fnames:
    print("Plotting %s" % fname)
    jk = JK.load_JK(fname)
    labels = ["%i" % rsep for rsep in jk.rsep]
    jk.corr.index = labels
    jk.corr.columns = labels
#plt.figure(figsize=[6, 6])
    sns.heatmap(jk.corr.iloc[:limitBins, :limitBins],
                vmin=-1, vmax=1, cmap='seismic',
                cbar_kws={"ticks": [-1, -0.5, 0.0, 0.5, 1.0]})
    plt.xlabel('r [Mpc]')
    plt.ylabel('r [Mpc]')
    if showTitle:
        plt.title(jk.name)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    plt.savefig(os.path.join(getPath(fname),
                             '%s.png' % jk.name),
                dpi=120)
    if show:
        plt.show()
    plt.close()

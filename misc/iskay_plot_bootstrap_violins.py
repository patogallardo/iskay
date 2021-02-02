#!/nfs/user/pag227/miniconda/bin/python
from iskay import JK
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import glob

fnames = glob.glob('results/*.pck')
assert len(fnames) > 0

for fname in fnames:
    jk = JK.load_JK(fname)
    N_bins_to_show = 10

    dfs = []
    for j in range(N_bins_to_show):
        dfs.append(pd.DataFrame({'p': jk.kSZ_curveJK_realizations[:, j],
                                 'rsep': jk.rsep[j]*np.ones(jk.JK_Ngroups)}))

    df = pd.concat(dfs, ignore_index=True)

    sns.violinplot(x="rsep", y="p", data=df, split=True)
    plt.xlabel('rsep [Mpc]')
    plt.ylabel('p [uK]')
    plt.title(jk.name)

    plt.axhline(0, color='black')
    plt.savefig('results/bootstrap_'+jk.name + '.png', dpi=150)
    plt.close()

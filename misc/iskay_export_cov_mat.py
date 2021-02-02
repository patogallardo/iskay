#!/nfs/user/pag227/miniconda/bin/python
'''This script exports jk data to csv files for further analysis.

Script should be run in the base directory, where results/ is present.
JK pck files should exist in results.

'''

from iskay import JK
import glob
import numpy as np
import pandas as pd
import os

assert os.path.exists('results_bs') or os.path.exists('results_jk')

fnames = glob.glob('results_*/*.pck')
fnames.sort()
print("Exporting cov mat and jk data in:")
print(fnames)

if os.path.exists('results_bs'):
    if not os.path.exists('results_bs/csv'):
        os.mkdir('results_bs/csv')
if os.path.exists('results_jk'):
    if not os.path.exists('results_jk/csv'):
        os.mkdir('results_jk/csv')
if os.path.exists('results_bsdt'):
    if not os.path.exists('results_bsdt/csv'):
        os.mkdir('results_bsdt/csv')


for fname in fnames:
    jk = JK.load_JK(fname)
    fname_out = '/'.join(fname.split('/')[:-1])
    fname_out = '%s/csv/%s' % (fname_out, jk.name)

    jk.corr.to_csv(fname_out + '_corr.csv')
    jk.cov.to_csv(fname_out + '_cov.csv')
    np.savetxt(fname_out + '_standard_error_realizations.csv',
               jk.kSZ_curveJK_realizations)

    data_and_errorbars = {'ksz_curve': jk.kSZ_curveFullDataset,
                          'rsep': jk.rsep,
                          'errobars': jk.errorbars}
    df = pd.DataFrame(data_and_errorbars)
    df.to_csv(fname_out + '_ksz_curve_and_errorbars.csv')

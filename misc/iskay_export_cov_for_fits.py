#!/nfs/user/pag227/miniconda/bin/python
'''Exports data made with export cov mat for a purely numeric format
to be read by curve fitting programs.'''
import pandas as pd
import glob
import numpy as np

fnames = glob.glob('results_*/csv/*cov.csv')

for fname in fnames:
    df = pd.read_csv(fname)
    mat = df.values[:, 1:]
    fname_out = fname[:-4] + '_rb.csv'
    np.savetxt(fname_out, mat)

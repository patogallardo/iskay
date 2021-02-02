#!/nfs/user/pag227/miniconda/bin/python
import glob
import pandas as pd
import numpy as np
from iskay import wiggly_tee_tools
from iskay import paramTools
import sys

sigma_z = 0.01
N_in_sigma = 20
gaussian_or_square = 'gaussian_conventional'
mean_or_median = 'mean'

assert len(sys.argv) == 2  # usage: correction paramfile.ini
p = paramTools.params(sys.argv[1])
filter_galaxies_by = p.CAT_QUERY # noqa


def wiggly_tee_correct(sigma_z, N_in_sigma,
                       gaussian_or_square,
                       mean_or_median,
                       filter_galaxies_by):
    fnames = glob.glob('ApPhotoResults/*.csv')
    fnames.sort()

    df = pd.concat([pd.read_csv(fname) for fname in fnames])
    df.query(filter_galaxies_by, inplace=True)
    # remmeber to edit this later

    df['ra_rad'] = np.deg2rad(df.ra.values)
    df['dec_rad'] = np.deg2rad(df.dec.values)

    print('Appliying wtee correction with:')
    print("sigma_z: %1.2f" % sigma_z)
    print('N_in_sigma: %i' % N_in_sigma)
    print("method: %s" % gaussian_or_square)
    print('mean_or_median: %s' % mean_or_median)
    print("filterBy: %s" % filter_galaxies_by)
    wiggly_tee_tools.correct_tsz_decrement(df, sigma_z,
            N_in_sigma=N_in_sigma,  # noqa
            gaussian_or_square=gaussian_or_square,  # noqa
            mean_or_median=mean_or_median)  # noqa
    df.to_csv('wtee_corrected_decrements_%s.csv' % p.NAME)


wiggly_tee_correct(sigma_z, N_in_sigma, gaussian_or_square,
                   mean_or_median, filter_galaxies_by)

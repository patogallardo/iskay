#!/nfs/user/pag227/miniconda/bin/python
'''Usage:
    arguments: ngroups, N_chnk, r_max
    N_chunk starts in 1 to make it compatible to sge.'''

import sys
from iskay import pairwise_and_save_data

import pandas as pd

N_groups, N_chunk = int(sys.argv[1]), int(sys.argv[2]) - 1
r_max = float(sys.argv[3])

df = pd.read_csv('wtee_corrected_decrements.csv')

pairwise_and_save_data.compute_one_pairwise_chunk_saving_everything_to_lnx1032(df, N_groups, N_chunk, r_max, MAX_NPAIRS=100000)  # noqa

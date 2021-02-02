#!/nfs/user/pag227/miniconda/bin/python
'''Usage:
    arguments: ngroups, N_chnk, r_max
    N_chunk starts in 1 to make it compatible to sge.

    N_groups: in how many chunks to split the computation
    N_chunk: what chunk to compute'''

import sys
from iskay import pairwise_and_save_data as pw_save
from iskay import paramTools
import pandas as pd

assert len(sys.argv) == 4
N_groups, N_chunk = int(sys.argv[1]), int(sys.argv[2]) - 1

params = paramTools.params(sys.argv[3])
df = pd.read_csv('wtee_corrected_decrements_%s.csv' % params.NAME)

pw_save.compute_one_pairwise_chunk_saving_everything_to_lnx1032(df, N_groups,
    N_chunk, params.BIN_EDGES, MAX_NPAIRS=100000)  # noqa

#! /nfs/user/pag227/miniconda/bin/python
''' pairwsise kSZ analysis script example for iskay.

Usage: iskay_analysis.py param.ini

Written by: P. Gallardo.
'''

from iskay import paramTools
from iskay import catalogTools
from iskay import JK
import sys

param_fname = sys.argv[1]
params = paramTools.params(param_fname)

df = catalogTools.preProcessedCat(howMany=None,
                                  query=params.CAT_QUERY).df
jk = JK.JK(df, params, distributed=True)

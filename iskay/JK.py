'''Container and setup for JK runs are located here.

For statistics and more JK amenities see JK_tools

Written by P. Gallardo'''
import numpy as np
from iskay import distributed_JK_kSZ
from iskay import singleMachine_JK_kSZ
from iskay import JK_tools
from iskay.JK_tools import getErrorbars
import pickle
import time
import os


def JK(df, params, distributed):
    jk = JK_container(df, params, distributed)
    save_JK(jk)
    return jk


def save_JK(jk_object):
    '''This pickles the entire JK structure.'''
    if not os.path.exists('results'):
        os.mkdir('results')
    fnameOut = os.path.join('./results/',
                            jk_object.params.NAME + '.pck')
    with open(fnameOut, 'w') as f:
        pickle.dump(jk_object, f)
    return 1


def load_JK(fname):
    '''Opens pickled JK container in fname.'''
    with open(fname, 'r') as f:
        jk = pickle.load(f)
    return jk


class JK_container():
    def __init__(self, df, params, distributed=True):
        self.params = params
        self.name = params.NAME
        self.query = params.CAT_QUERY
        self.do_variance_weighted = params.DO_VARIANCE_WEIGHTED
        self.N_objects_in_this_run = len(df)
        self.JK_Ngroups = params.JK_NGROUPS
        self.runJK(df, self.params, distributed)
        self.cov = JK_tools.getCovMatrix(self.bin_names,
                                         self.kSZ_curveJK_realizations)
        self.corr = JK_tools.getCorrMatrix(self.bin_names,
                                           self.kSZ_curveJK_realizations)

    def runJK(self, df, params, distributed):
        t1 = time.time()
        if distributed is True:
            res = distributed_JK_kSZ.run_JK_distributed(df, params,
                                                        randomize=True)
        else:
            res = singleMachine_JK_kSZ.run_JK_local(df, params,
                                                    randomize=True)
        t2 = time.time()
        fullDataset_results, jk_results = res
        rsep = fullDataset_results[0]
        p_uk = fullDataset_results[1]

        jk_results = [jk_results[j][1] for j in range(len(jk_results))]
        jk_results = np.array(jk_results)
        self.rsep = rsep
        self.bin_names = JK_tools.getBinNames(rsep)
        self.kSZ_curveFullDataset = p_uk
        self.kSZ_curveJK_realizations = jk_results
        self.errorbars = getErrorbars(jk_results, params)
        self.runtime = t2 - t1

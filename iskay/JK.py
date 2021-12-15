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
from pairwiser_massboosted import run_JK_distributed_massboosted


def JK(df, params, distributed):
    jk = JK_container(df, params, distributed)
    save_JK(jk)
    return jk


def save_JK(jk_object):
    '''This pickles the entire JK structure.'''
    if jk_object.params.JK_RESAMPLING_METHOD.lower() == 'jk':
        output_dir = 'results_jk'
    elif jk_object.params.JK_RESAMPLING_METHOD.lower() == 'bootstrap':
        output_dir = 'results_bs'
    elif jk_object.params.JK_RESAMPLING_METHOD.lower() == 'bootstrap_pairwise':
        output_dir = 'results_bs_pairwise'
    elif jk_object.params.JK_RESAMPLING_METHOD.lower() == 'bs_dt':
        output_dir = 'results_bsdt'
    elif jk_object.params.JK_RESAMPLING_METHOD.lower() == 'tiled_jk':
        output_dir = 'results_tiled_jk'
    elif jk_object.params.JK_RESAMPLING_METHOD.lower() == 'bs_dt_mass_boosted_est':  # noqa
        output_dir = 'results_bsdt_mass_boosted_est'
    elif jk_object.params.JK_RESAMPLING_METHOD.lower() == 'bs_dt_mass_boosted_est_debiased':  # noqa
        output_dir = 'results_bsdt_mass_boosted_est_debiased'
    else:
        assert False  # unknown resampling method
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fnameOut = os.path.join('./%s' % output_dir,
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
        if 'tiled' in self.params.JK_RESAMPLING_METHOD:
            self.JK_Ngroups = self.kSZ_curveJK_realizations.shape[0]
        self.cov = JK_tools.getCovMatrix(self.bin_names,
                                         self.kSZ_curveJK_realizations,
                                         params)
        self.corr = JK_tools.getCorrMatrix(self.bin_names,
                                           self.kSZ_curveJK_realizations)

    def runJK(self, df, params, distributed):
        t1 = time.time()
        if distributed is True:
            resampling_method = params.JK_RESAMPLING_METHOD.lower()
            do_massboosted = resampling_method == 'bs_dt_mass_boosted_est'
            do_massboosted_debiased = resampling_method == 'bs_dt_mass_boosted_est_debiased'  # noqa
            if do_massboosted or do_massboosted_debiased:
                res = run_JK_distributed_massboosted(df, params)  # noqa
            else:
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
        self.bin_edges = params.BIN_EDGES
        self.bin_names = JK_tools.getBinNamesFromBinEdges(params.BIN_EDGES)
        self.kSZ_curveFullDataset = p_uk
        self.kSZ_curveJK_realizations = jk_results
        self.errorbars = getErrorbars(jk_results, params)
        self.runtime = t2 - t1

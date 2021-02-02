'''Container and setup for JK runs are located here.

For statistics and more JK amenities see JK_tools

Written by P. Gallardo'''
import numpy as np
from iskay import cross_distributed
from iskay import JK_tools
import pickle
import time
import os


def resampled(ds, params):
    rs = resampled_container(ds, params)
    save_resampled(rs)
    return rs


def save_resampled(resampled_object):
    '''This pickles the entire resampled_container structure.'''
    if resampled_object.params.JK_RESAMPLING_METHOD.lower() == 'jk':
        output_dir = 'results_jk'
    elif resampled_object.params.JK_RESAMPLING_METHOD.lower() == 'bootstrap':
        output_dir = 'results_bs'
    else:
        assert False  # unknown resampling method
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    fnameOut = os.path.join('./%s' % output_dir,
                            resampled_object.params.NAME + '.pck')
    with open(fnameOut, 'w') as f:
        pickle.dump(resampled_object, f)
    return 1


def load_resampled(fname):
    '''Opens pickled JK container in fname.'''
    with open(fname, 'r') as f:
        resampled_object = pickle.load(f)
    return resampled_object


class resampled_container():
    def __init__(self, ds, params):
        df1, df2 = ds.df1, ds.df2
        self.params = params
        self.name = params.NAME
        self.query = params.CAT_QUERY
        self.do_variance_weighted = params.DO_VARIANCE_WEIGHTED
        self.N_objects_in_this_run = len(df1)
        self.JK_Ngroups = params.JK_NGROUPS
        self.run_resample(df1, df2, self.params)

        self.cov11 = JK_tools.getCovMatrix(self.bin_names,
                                           self.kSZ_curveJK_realizations11,
                                           params)
        self.cov12 = JK_tools.getCovMatrix(self.bin_names,
                                           self.kSZ_curveJK_realizations12,
                                           params)
        self.cov22 = JK_tools.getCovMatrix(self.bin_names,
                                           self.kSZ_curveJK_realizations22,
                                           params)

        self.corr11 = JK_tools.getCorrMatrix(self.bin_names,
                                             self.kSZ_curveJK_realizations11)
        self.corr12 = JK_tools.getCorrMatrix(self.bin_names,
                                             self.kSZ_curveJK_realizations12)
        self.corr22 = JK_tools.getCorrMatrix(self.bin_names,
                                             self.kSZ_curveJK_realizations22)

    def run_resample(self, df1, df2, params):
        t1 = time.time()
        res = cross_distributed.run_error_estimation_distributed(df1, df2,
                                                                 params)
        t2 = time.time()

        fullDataset_results11 = res['full11']
        fullDataset_results12 = res['full12']
        fullDataset_results22 = res['full22']

        rsep = fullDataset_results11[0]
        p_uk11 = fullDataset_results11[1]
        p_uk12 = fullDataset_results12[1]
        p_uk22 = fullDataset_results22[1]

        resampled_results11 = [res['resampled11'][j][1]
                               for j in range(params.JK_NGROUPS)]
        resampled_results12 = [res['resampled12'][j][1]
                               for j in range(params.JK_NGROUPS)]
        resampled_results22 = [res['resampled22'][j][1]
                               for j in range(params.JK_NGROUPS)]

        resampled_results11 = np.array(resampled_results11)
        resampled_results12 = np.array(resampled_results12)
        resampled_results22 = np.array(resampled_results22)

        self.rsep = rsep
        self.bin_edges = params.BIN_EDGES
        self.bin_names = JK_tools.getBinNamesFromBinEdges(params.BIN_EDGES)

        self.kSZ_curveFullDataset11 = p_uk11
        self.kSZ_curveFullDataset12 = p_uk12
        self.kSZ_curveFullDataset22 = p_uk22

        self.kSZ_curveJK_realizations11 = resampled_results11
        self.kSZ_curveJK_realizations12 = resampled_results12
        self.kSZ_curveJK_realizations22 = resampled_results22

        self.errorbars11 = JK_tools.getErrorbars(resampled_results11, params)
        self.errorbars12 = JK_tools.getErrorbars(resampled_results12, params)
        self.errorbars22 = JK_tools.getErrorbars(resampled_results22, params)

        self.runtime = t2 - t1

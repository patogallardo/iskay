'''
Tools to run the ksz estimator over a cluster using Dask.

Written by P. Gallardo
'''
from iskay import envVars
from iskay import pairwiser
from iskay import bootstrap_pairwise as bs_pw
from iskay import tiled_JK
from dask.distributed import Client  #, progress
from dask_jobqueue import SGECluster
from iskay import JK_tools
import time
import numpy as np

JK = 'jk'
TL_JK = 'tiled_jk'
BS = 'bootstrap'
BS_PW = 'bootstrap_pairwise'
BS_DT = 'bs_dt'


def run_JK_distributed(df, param, randomize=True):
    '''Receives the pandas dataframe with the objects containing the
    temperature decrements and the parameter object and run the kSZ
    statistic and generate Jack Knifes.
    Everything runs in the cluster, so current terminal does not need
    to request many cpus.

    df: dataframe object containing the variables for the calculation
    params: param file for this calculation
    NJK: how many subgroups we will make to run the calculation
    randomize: shuffle data before running the JK'''

    Ncores = envVars.Ncores
    NWorkers = envVars.NWorkers
    Ngroups = param.JK_NGROUPS
    resampling_method = param.JK_RESAMPLING_METHOD.lower()

    #setup cluster
    cluster = SGECluster(walltime='172800', processes=1, cores=1,
                         env_extra=['#$-pe sge_pe %i' % Ncores,
                                    '-l m_core=%i' % Ncores,
                                    'mkdir -p /tmp/pag227/dask/dask-scratch',
                                    'export NUMBA_NUM_THREADS=%i' % Ncores,
                                    'export OMP_NUM_THREADS=%i' % Ncores
#                                    'export OMP_NUM_THREADS=1',  # noqa
                                    ])
    cluster.scale(NWorkers)
    client = Client(cluster)
    time.sleep(30)
    #end setting up cluster

    #send full dataset to the cluster
    future_fullDataset = client.scatter(df)
    future_params = client.scatter(param)
    res_fullDataset = client.submit(pairwiser.get_pairwise_ksz,
                                    future_fullDataset,
                                    future_params, multithreading=True)
    #done with the full dataset

    #iterate over partial dataset for the JK
    if JK == resampling_method:
        indices_toDrop = JK_tools.indicesToDrop(df, Ngroups,
                                                randomize=randomize)
    jk_results = []
    futureData = []  #data to be sent in jk or bootstrap in galaxy space

    if (JK == resampling_method) or (BS == resampling_method):
        for j in range(Ngroups):  # submit data to the cluster
            if JK in resampling_method:  # if method jk
                dataJK = df.drop(indices_toDrop[j], inplace=False)
                futureData.append(client.scatter(dataJK))
            elif BS in resampling_method:
                dataBS = df.sample(len(df), replace=True)
                futureData.append(client.scatter(dataBS))
        #Now do the JK calculation
        for j in range(Ngroups):
            jk_results.append(client.submit(pairwiser.get_pairwise_ksz,
                              futureData[j],
                              future_params, multithreading=True))

    if BS_PW == resampling_method:  # submit the same dataset
        futureData = client.scatter(df, broadcast=True)

        for j in range(Ngroups):
            jk_results.append(client.submit(bs_pw.get_bootstrap_pairwise,
                                            futureData,
                                            future_params,
                                            multithreading=True,
                                            pure=False))
    if resampling_method == BS_DT:
        for j in range(Ngroups):
            df_bs = df.copy()
            choose = np.random.choice(len(df), len(df))
            df_bs['dT'] = df.dT.values[choose]
            futureData.append(client.scatter(df_bs))
        for j in range(Ngroups):
            jk_results.append(client.submit(pairwiser.get_pairwise_ksz,
                                            futureData[j],
                                            future_params,
                                            multithreading=True))

    if resampling_method == TL_JK:
        tiled_JK.classify_grid(df)
        df = tiled_JK.remove_edge_galaxies(df, tol_sigma=1.5)
        Ntiles = tiled_JK.how_many_tiles(df)
        for j in range(Ntiles):
            df_tosubmit = tiled_JK.remove_tile(df, j)
            futureData.append(client.scatter(df_tosubmit))
        for j in range(Ntiles):
            jk_results.append(client.submit(pairwiser.get_pairwise_ksz,
                                            futureData[j],
                                            future_params,
                                            multithreading=True))
    #extract results
    fullDataset_results = res_fullDataset.result()
    jk_results = client.gather(jk_results)
    client.close()
#    cluster.close()

    return fullDataset_results, jk_results

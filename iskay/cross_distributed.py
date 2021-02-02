'''
Runs the cross ksz estimator on a cluster and iterates to find
covariances.
'''

import numpy as np
from iskay import envVars
from iskay import cross_ksz_pairwiser as cpw
from dask.distributed import Client
from dask_jobqueue import SGECluster
import time


def run_error_estimation_distributed(df1, df2, param):
    Ncores = envVars.Ncores
    NWorkers = envVars.NWorkers
    Ngroups = param.JK_NGROUPS

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
    time.sleep(10)
    #end setting up cluster

    #send full dataset to the cluster
    future_df1 = client.scatter(df1)
    future_df2 = client.scatter(df2)

    future_params = client.scatter(param)
    res_fullDataset_11 = client.submit(cpw.get_cross_pairwise_ksz,
                                       future_df1, future_df1,
                                       future_params)
    res_fullDataset_12 = client.submit(cpw.get_cross_pairwise_ksz,
                                       future_df1, future_df2,
                                       future_params)
    res_fullDataset_22 = client.submit(cpw.get_cross_pairwise_ksz,
                                       future_df2, future_df2,
                                       future_params)
    #done with the full dataset

    #iterate over partial dataset for the JK
    replicants1 = []  #data to be sent
    replicants2 = []

    if 'jk' in param.JK_RESAMPLING_METHOD.lower():
        all_indx = np.arange(len(df1))
        np.random.shuffle(all_indx)
        indx_to_drop = np.array_split(all_indx, param.JK_NGROUPS)
    for j in range(Ngroups):  # submit data to the cluster
        if 'jk' in param.JK_RESAMPLING_METHOD.lower():  # if method jk
            todrop = indx_to_drop[j]
            replicant1 = df1.drop(df1.index[todrop], inplace=False)
            replicant2 = df2.drop(df2.index[todrop], inplace=False)

            replicants1.append(client.scatter(replicant1))
            replicants2.append(client.scatter(replicant2))
        elif 'bootstrap' in param.JK_RESAMPLING_METHOD.lower():
            indxs = np.random.randint(low=0, high=len(df1),
                                      size=len(df1))
            replicant1 = df1.iloc[indxs]
            replicant2 = df2.iloc[indxs]
            replicants1.append(client.scatter(replicant1))
            replicants2.append(client.scatter(replicant2))

    #Now do the JK calculation
    realizations11 = []
    realizations12 = []
    realizations22 = []

    for j in range(Ngroups):
        realizations11.append(client.submit(cpw.get_cross_pairwise_ksz,
                              replicants1[j], replicants1[j],
                              future_params))
        realizations12.append(client.submit(cpw.get_cross_pairwise_ksz,
                              replicants1[j], replicants2[j],
                              future_params))
        realizations22.append(client.submit(cpw.get_cross_pairwise_ksz,
                              replicants2[j], replicants2[j],
                              future_params))
    #extract results
    fullDataset_result11 = res_fullDataset_11.result()
    fullDataset_result12 = res_fullDataset_12.result()
    fullDataset_result22 = res_fullDataset_22.result()

    resampling_result11 = client.gather(realizations11)
    resampling_result12 = client.gather(realizations12)
    resampling_result22 = client.gather(realizations22)
    client.close()
#    cluster.close()

    results = {'full11': fullDataset_result11,
               'full12': fullDataset_result12,
               'full22': fullDataset_result22,
               'resampled11': resampling_result11,
               'resampled12': resampling_result12,
               'resampled22': resampling_result22}

    return results

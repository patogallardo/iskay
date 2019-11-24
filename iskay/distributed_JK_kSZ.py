'''
Tools to run the ksz estimator over a cluster using Dask.

Written by P. Gallardo
'''
from iskay import envVars
from iskay import pairwiser
from dask.distributed import Client  #, progress
from dask_jobqueue import SGECluster
from iskay import JK_tools
import time


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

    #setup cluster
    cluster = SGECluster(walltime='172800', processes=1, cores=1,
                         env_extra=['#$-pe sge_pe %i' % Ncores,
                                    '-l m_core=%i' % Ncores,
                                    'mkdir -p /tmp/pag227/dask/dask-scratch',
                                    'export NUMBA_NUM_THREADS=%i' % Ncores,
                                    'export OMP_NUM_THREADS=%i' % Ncores
#                                    'export OMP_NUM_THREADS=1',
                                    ])
    cluster.scale(NWorkers)
    client = Client(cluster)
    time.sleep(10)
    #end setting up cluster

    #send full dataset to the cluster
    future_fullDataset = client.scatter(df)
    future_params = client.scatter(param)
    res_fullDataset = client.submit(pairwiser.get_pairwise_ksz,
                                    future_fullDataset,
                                    future_params, multithreading=True)
    #done with the full dataset

    #iterate over partial dataset for the JK
    indices_toDrop = JK_tools.indicesToDrop(df, Ngroups, randomize=randomize)
    futureData = []  #data to be sent
    for j in range(Ngroups):  # submit data to the cluster
        dataJK = df.drop(indices_toDrop[j], inplace=False)
        futureData.append(client.scatter(dataJK))

    #Now do the JK calculation
    jk_results = []
    for j in range(Ngroups):
        jk_results.append(client.submit(pairwiser.get_pairwise_ksz,
                          futureData[j],
                          future_params, multithreading=True))
    #extract results
    fullDataset_results = res_fullDataset.result()
    jk_results = client.gather(jk_results)
    client.close()
    cluster.close()

    return fullDataset_results, jk_results

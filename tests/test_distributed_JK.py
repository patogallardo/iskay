from iskay import distributed_JK_kSZ
from iskay import singleMachine_JK_kSZ
import numpy as np
from iskay import cosmology
import pandas as pd
import os
from iskay import paramTools
from iskay import catalogTools


def test_distributed_JK():
    '''This tests the distributed jk against the single machine jk.'''
    df = produceFakeCatalog(7000)
    params = produceFakeParams()

    res_local = singleMachine_JK_kSZ.run_JK_local(df, params,
                                                  randomize=False)
    res_distributed = distributed_JK_kSZ.run_JK_distributed(df,
                                                            params,
                                                            randomize=False)
    res_local_complete = res_local[0]
    res_local_JK = res_local[1]

    res_distributed_complete = res_distributed[0]
    res_distributed_JK = res_distributed[1]

    err_sq = np.sum((res_local_complete[0] - res_distributed_complete[0])**2)
    assert err_sq < 1e-10

    err_sq = np.sum((res_local_complete[1] - res_distributed_complete[1])**2)
    assert err_sq < 1e-10

    for j in range(len(res_local_JK)):
        err_sq = np.sum((res_local_JK[j][0] - res_distributed_JK[j][0])**2)
        assert err_sq < 1e-10
        err_sq = np.sum((res_local_JK[j][1] - res_distributed_JK[j][1])**2)
        assert err_sq < 1e-10


def produceFakeCatalog(Nobj=10000):
    ''' Returns a fake pandas dataframe with data for pairwiser_ksz'''

    z = np.random.uniform(0, 1, Nobj)
    Dc = cosmology.Dc(z)
    ra_deg = np.random.uniform(0, 350, Nobj)
    dec_deg = np.random.uniform(-30, 0, Nobj)
    dT = np.random.uniform(-300, 300, Nobj)
    datain = {'z': z, 'Dc': Dc, 'ra': ra_deg, 'dec': dec_deg, 'dT': dT}
    df = pd.DataFrame(datain)
    return df


def produceFakeParams():
    testPath = '/'.join((catalogTools.__file__).split('/')[:-2]) + '/tests/'
    testParamFileFullPath = os.path.join(testPath, 'data_toTestAPI/params.ini')
    params = paramTools.params(testParamFileFullPath)
    return params

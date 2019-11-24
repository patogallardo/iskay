from test_distributed_JK import produceFakeCatalog
from test_distributed_JK import produceFakeParams
import numpy as np
from iskay import JK


def test_JK_container():
    df = produceFakeCatalog()
    params = produceFakeParams()
    params.JK_NGROUPS = 3
    jk = JK.JK_container(df, params, distributed=False)
    diff_sq = np.sum((np.diag(jk.cov) - jk.errorbars**2)**2)
    assert diff_sq < 1e10

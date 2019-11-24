'''Idem to istributed_JK_kSZ.py but runs on a huge for loop locally.

Make sure to allocate all the cores you will need.
'''
from iskay import JK_tools
from iskay import pairwiser


def run_JK_local(df, params, randomize=True, multithreading=False):
    '''Receives the pandas df with objects with temp decrements and the
    parameter file object.

    Runs the ksz estimator and runs jackknifes.

    Everything runs locally, make sure you have requested the resources you
    are using.

    df: dataframe object with the variables for the calculation
    params: param file for this calculation
    NJK: how many subgroups for the run_JK
    '''
    print("Running a JK run on the local machine, this will take a while.")
    Ngroups = params.JK_NGROUPS
    fullDataset_results = pairwiser.get_pairwise_ksz(df,
                                                     params,
                                  multithreading=multithreading) # noqa
    indices_toDrop = JK_tools.indicesToDrop(df, Ngroups, randomize=randomize)
    jk_results = []
    for j in range(Ngroups):
        print "%i/%i" % (j, Ngroups)
        data_JK = df.drop(indices_toDrop[j], inplace=False)
        jk_results.append(pairwiser.get_pairwise_ksz(data_JK,
                                                     params,
                                    multithreading=multithreading)) # noqa
    return fullDataset_results, jk_results

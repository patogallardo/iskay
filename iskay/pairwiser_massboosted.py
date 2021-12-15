'''This module contains the core pairwise kSZ estimator algorithm.
  '''
import numba
import math as mt
from iskay.pairwiser import inWhatBinIsIt, angle_jit_rad, vecdiff_jit
from iskay.pairwiser import get_tzav_fast, make_rsep_uneven_bins
import numpy as np
import itertools
from concurrent import futures
from iskay import envVars
from dask.distributed import Client
from dask_jobqueue import SGECluster
import time


@numba.jit(nopython=True, nogil=True)
def pairwiser_massboost_one_row(row, Dc, ra_rad, dec_rad, tzav,
                                Tmapsc, bin_edges, mass, dTw, w2, sum_mass,
                                Npairs):
    '''
    inputs:
    row: what row to compute
    Dc: distances
    ra_rad: ra
    dec_rad: dec
    tzav: averaged T vs z
    Tmapsc: Tdisc-Tring
    bin_edges: bin edges in Mpc (for n bins, we need n+1)
    mass: mass in linear scale

    outputs:
    dTw: accumulated sum dT * c_ij * M_ij
    w2: accumulated sum c_ij**2 * M_ij**2
    sum_mass: sum M_ij
    Npairs: counter'''
    many = len(ra_rad)
    i = row
    for j in range(i+1, many):
        ang_ij = angle_jit_rad(dec_rad[i], ra_rad[i], dec_rad[j], ra_rad[j])
        vecdiff_ij = vecdiff_jit(Dc[i], Dc[j], ang_ij)
        if (vecdiff_ij > bin_edges[0]) and (vecdiff_ij < bin_edges[-1]):
            binval_ij = inWhatBinIsIt(vecdiff_ij, bin_edges)
            dT_ij = (Tmapsc[i] - tzav[i])-(Tmapsc[j]-tzav[j])
            cij = (Dc[i]-Dc[j])*(1.0 + mt.cos(ang_ij)) / (2.0*vecdiff_ij)
            Mij = (mass[i] + mass[j])/2.
            dTw[binval_ij] += dT_ij * cij * Mij
            w2[binval_ij] += cij**2. * Mij**2
            sum_mass[binval_ij] += Mij
            Npairs[binval_ij] += 1


@numba.jit(nopython=True, nogil=True)
def pairwiser_massboost_moments_one_row(row, Dc, ra_rad, dec_rad, tzav,
                                        Tmapsc, bin_edges, mass,
                                        dTw, w2, sum_T, sum_c,
                                        sum_mass, Npairs):
    '''
    inputs:
    row: what row to compute
    Dc: distances
    ra_rad: ra
    dec_rad: dec
    tzav: averaged T vs z
    Tmapsc: Tdisc-Tring
    bin_edges: bin edges in Mpc (for n bins, we need n+1)
    mass: mass in linear scale

    outputs:
    dTw: accumulated sum dT * c_ij * M_ij
    w2: accumulated sum c_ij**2 * M_ij**2
    sum_mass: sum M_ij
    Npairs: counter'''
    many = len(ra_rad)
    i = row
    for j in range(i+1, many):
        ang_ij = angle_jit_rad(dec_rad[i], ra_rad[i], dec_rad[j], ra_rad[j])
        vecdiff_ij = vecdiff_jit(Dc[i], Dc[j], ang_ij)
        if (vecdiff_ij > bin_edges[0]) and (vecdiff_ij < bin_edges[-1]):
            binval_ij = inWhatBinIsIt(vecdiff_ij, bin_edges)
            dT_ij = (Tmapsc[i] - tzav[i])-(Tmapsc[j]-tzav[j])  # minus tau

            cij = (Dc[i]-Dc[j])*(1.0 + mt.cos(ang_ij)) / (2.0*vecdiff_ij)
            Mij = (mass[i] + mass[j])/2.
            cij = Mij * cij

            dTw[binval_ij] += dT_ij * cij
            w2[binval_ij] += cij**2.
            sum_T[binval_ij] += dT_ij
            sum_c[binval_ij] += cij
            sum_mass[binval_ij] += Mij
            Npairs[binval_ij] += 1


def pairwise_ksz_massboosted(Dc, ra_deg, dec_deg, tzav, Tmapsc, mass,
                             bin_edges,
                             Nthreads=numba.config.NUMBA_NUM_THREADS):
    assert len(Dc) == len(ra_deg) == len(dec_deg) == len(tzav) == len(Tmapsc)
    assert len(Dc) == len(mass)

    nrbin = len(bin_edges) - 1
    length = len(ra_deg)

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    dTws = [np.zeros(nrbin) for j in range(length-1)]
    w2s = [np.zeros(nrbin) for j in range(length-1)]
    Npairs = [np.zeros(nrbin, dtype=int) for j in range(length-1)]
    Msums = [np.zeros(nrbin) for j in range(length-1)]

    rows = xrange(length-1)
    Dcs = itertools.repeat(Dc, length-1)
    ras_rad = itertools.repeat(ra_rad, length-1)
    decs_rad = itertools.repeat(dec_rad, length-1)
    tzavs = itertools.repeat(tzav, length-1)
    Tmapscs = itertools.repeat(Tmapsc, length-1)
    bin_edges_s = itertools.repeat(bin_edges, length-1)
    masses = itertools.repeat(mass, length-1)

    with futures.ThreadPoolExecutor(Nthreads) as ex:
        ex.map(pairwiser_massboost_one_row,
               rows, Dcs, ras_rad, decs_rad, tzavs, Tmapscs,
               bin_edges_s, masses,
               dTws, w2s, Msums, Npairs)

    dTws = np.array(dTws)
    w2s = np.array(w2s)
    Msums = np.array(Msums)
    Npairs = np.array(Npairs)

    dTw = dTws.sum(axis=0)
    w2 = w2s.sum(axis=0)
    Msum = Msums.sum(axis=0)
    Np = Npairs.sum(axis=0)

    assert not np.any(w2 == 0)

    pest = -dTw/w2 * (Msum/Np)
    rsep = make_rsep_uneven_bins(bin_edges)

    return rsep, pest


def pairwise_ksz_massboosted_debiased(Dc, ra_deg, dec_deg, tzav, Tmapsc, mass,
                                      bin_edges,
                       Nthreads = numba.config.NUMBA_NUM_THREADS):  # noqa
    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    nrbin = len(bin_edges) - 1
    length = len(ra_deg)

    dTws = [np.zeros(nrbin) for j in range(length-1)]
    w2s = [np.zeros(nrbin) for j in range(length-1)]
    Npairs = [np.zeros(nrbin, dtype=int) for j in range(length-1)]
    Msums = [np.zeros(nrbin) for j in range(length-1)]
    sum_c = [np.zeros(nrbin) for j in range(length-1)]
    sum_T = [np.zeros(nrbin) for j in range(length-1)]

    rows = xrange(length-1)
    Dcs = itertools.repeat(Dc, length-1)
    ras_rad = itertools.repeat(ra_rad, length-1)
    decs_rad = itertools.repeat(dec_rad, length-1)
    tzavs = itertools.repeat(tzav, length-1)
    Tmapscs = itertools.repeat(Tmapsc, length-1)
    bin_edges_s = itertools.repeat(bin_edges, length-1)
    masses = itertools.repeat(mass, length-1)

    with futures.ThreadPoolExecutor(Nthreads) as ex:
        ex.map(pairwiser_massboost_moments_one_row,
               rows, Dcs, ras_rad, decs_rad, tzavs, Tmapscs,
               bin_edges_s, masses,
               dTws, w2s, sum_T, sum_c, Msums, Npairs)
    dTws = np.array(dTws)
    w2s = np.array(w2s)
    Msums = np.array(Msums)
    Npairs = np.array(Npairs)
    sum_c = np.array(sum_c)
    sum_T = np.array(sum_T)

    Np = Npairs.sum(axis=0)
    dTw = dTws.sum(axis=0)/Np
    w2 = w2s.sum(axis=0)/Np
    Mbar = Msums.sum(axis=0)/Np
    c_bar = sum_c.sum(axis=0)/Np
    T_bar = sum_T.sum(axis=0)/Np

    num = (dTw - T_bar * c_bar) * Mbar
    den = (w2 - c_bar**2)
    pest = -num/den

    rsep = make_rsep_uneven_bins(bin_edges)

    return rsep, pest


def run_JK_distributed_massboosted(df, param):
    '''Receives the pandas dataframe with the objects containing the
    temperature decrements and the parameter object and run the kSZ
    statistic and generate Jack Knifes.
    Everything runs in the cluster, so current terminal does not need
    to request many cpus.

    df: dataframe object containing the variables for the calculation
    params: param file for this calculation
    NJK: how many subgroups we will make to run the calculation'''

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
    time.sleep(30)
    #end setting up cluster

    #send full dataset to the cluster
    future_fullDataset = client.scatter(df)
    future_params = client.scatter(param)
    res_fullDataset = client.submit(get_pairwise_ksz_massboosted,
                                    future_fullDataset,
                                    future_params, multithreading=True)
    #done with the full dataset
    jk_results = []
    futureData = []  #data to be sent in jk or bootstrap in galaxy space

    for j in range(Ngroups):
        df_bs = df.copy()
        choose = np.random.choice(len(df), len(df))
        df_bs['dT'] = df.dT.values[choose]
        futureData.append(client.scatter(df_bs))

    if param.JK_RESAMPLING_METHOD.lower() == "bs_dt_mass_boosted_est":
        get_pw_func = get_pairwise_ksz_massboosted
    elif param.JK_RESAMPLING_METHOD.lower() == 'bs_dt_mass_boosted_est_debiased':  # noqa
        get_pw_func = get_pairwise_ksz_massboosted_debiased

    for j in range(Ngroups):
        jk_results.append(client.submit(get_pw_func,
                                        futureData[j],
                                        future_params,
                                        multithreading=True))
# extract results
    fullDataset_results = res_fullDataset.result()
    jk_results = client.gather(jk_results)
    client.close()
#  cluster.close()

    return fullDataset_results, jk_results


def get_pairwise_ksz_massboosted(df, params,
                                 multithreading=True):
    '''Wrapper for calling pairwise_ksz using only a preprocessed dataframe
    and a parameters container.
    Arguments:
        df: preprocessed catalog
        params: paramTols.params object.'''
    dT = df.dT.values
    z = df.z.values
    Dc = df.Dc.values
    mass = df.mass.values

    tzav = get_tzav_fast(dT, z, params.SIGMA_Z)  # use fast implementation
    ra_deg = df.ra.values
    dec_deg = df.dec.values
    bin_edges = params.BIN_EDGES

    r_sep, p_uk = pairwise_ksz_massboosted(Dc, ra_deg, dec_deg, tzav,
                                           dT, mass, bin_edges)
    return r_sep, p_uk


def get_pairwise_ksz_massboosted_debiased(df, params,
                                          multithreading=True):
    '''Wrapper for calling pairwise_ksz using only a preprocessed dataframe
    and a parameters container.
    Arguments:
        df: preprocessed catalog
        params: paramTols.params object.'''
    dT = df.dT.values
    z = df.z.values
    Dc = df.Dc.values
    mass = df.mass.values

    tzav = get_tzav_fast(dT, z, params.SIGMA_Z)  # use fast implementation
    ra_deg = df.ra.values
    dec_deg = df.dec.values
    bin_edges = params.BIN_EDGES

    r_sep, p_uk = pairwise_ksz_massboosted_debiased(Dc, ra_deg, dec_deg, tzav,
                                                    dT, mass, bin_edges)
    return r_sep, p_uk

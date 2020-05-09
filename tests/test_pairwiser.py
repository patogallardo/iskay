import numpy as np
import numba
import math as mt
import iskay.pairwiser as pairwiser
from iskay import paramTools
import itertools
from concurrent import futures
from iskay import catalogTools
import os


def test_inWhatBinIsIt():
    val = 5.4
    bin_edges = np.array([3, 4, 6, 8, 10, 14, 17.], dtype=float)
    assert pairwiser.inWhatBinIsIt(val, bin_edges) == 1


def test_pairwiser_one_row_uneven_bins():
    length = 10000
    row = np.random.randint(0, length)
    Dc = np.random.uniform(low=100, high=110, size=length)
    ra_rad = np.random.uniform(low=0, high=2*np.pi, size=length)
    dec_rad = np.random.uniform(low=-30, high=30, size=length)
    tzav = np.zeros(length)
    Tmapsc = np.random.uniform(low=0, high=20, size=length)
    nrbin = 40
    binsz = 5
    bin_edges = np.arange(0, binsz*(nrbin+1.0), binsz)
    dTw_pairwise_one_row = np.zeros(nrbin)
    w2_pairwise_one_row = np.zeros(nrbin)
    dTw_pariwise_one_row_unevenBins = np.zeros(nrbin)
    w2_pairwise_one_row_unevenBins = np.zeros(nrbin)
    pairwiser.pairwise_one_row(row, Dc, ra_rad, dec_rad, tzav,
                               Tmapsc, nrbin, binsz, dTw_pairwise_one_row,
                               w2_pairwise_one_row)
    pairwiser.pairwise_one_row_uneven_bins(row, Dc, ra_rad, dec_rad, tzav,
                                           Tmapsc, bin_edges,
                                           dTw_pariwise_one_row_unevenBins,
                                           w2_pairwise_one_row_unevenBins)
    sum_err_sq1 = np.sum((dTw_pairwise_one_row - # noqa
                          dTw_pariwise_one_row_unevenBins)**2)
    sum_err_sq2 = np.sum((w2_pairwise_one_row -  # noqa
                          w2_pairwise_one_row_unevenBins)**2)
    assert sum_err_sq1 < 1e-10 and sum_err_sq2 < 1e-10


def test_get_tzav():
    size = 1000
    dTs = np.random.uniform(low=0, high=100, size=size)
    zs = np.random.uniform(low=0, high=100, size=size)
    dTs = dTs + zs

    sigma_z = 10

    numer = np.empty_like(dTs)
    denom = np.empty_like(dTs)
    for j in range(size):
        numer[j] = np.sum(dTs*np.exp(-(zs[j]-zs)**2/(2*sigma_z**2)))
        denom[j] = np.sum(np.exp(-(zs[j]-zs)**2/(2*sigma_z**2)))
    Tz_numpy = numer/denom

    Tz_numba = pairwiser.get_tzav(dTs, zs, sigma_z)

    diff_sq = (Tz_numpy-Tz_numba)**2
    error = diff_sq.sum()
    assert error < 1e-10


def test_vecdiff_jit():
    size = 20000
    d1 = np.random.uniform(low=0.1, high=500, size=size)
    d2 = np.random.uniform(low=0.1, high=500, size=size)
    angles_rd = np.random.uniform(low=0, high=2*np.pi, size=size)

    vecdiff_pairwiser = [pairwiser.vecdiff_jit(dist1, dist2, ang) for
                         dist1, dist2, ang in zip(d1, d2, angles_rd)]
    vecdiff_pairwiser = np.array(vecdiff_pairwiser)
    vecdiff = np.sqrt(d1**2 + d2**2 - 2*d1*d2*np.cos(angles_rd))
    diff_sq = (vecdiff_pairwiser - vecdiff)**2
    assert diff_sq.sum() < 1e-10


def test_angle_jit_deg():
    size = 20000
    lat1 = np.random.uniform(low=-90, high=90, size=size)
    lat2 = np.random.uniform(low=-90, high=90, size=size)
    long1 = np.random.uniform(low=0, high=360, size=size)
    long2 = np.random.uniform(low=0, high=360, size=size)

    angles_jit_deg = np.array(
        map(angle_jit_deg, lat1, long1, lat2, long2))

    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    long1_rad = np.deg2rad(long1)
    long2_rad = np.deg2rad(long2)
    angles_np = np.arccos(np.sin(lat1_rad)*np.sin(lat2_rad) +  # noqa
                        np.cos(lat1_rad)*np.cos(lat2_rad)* # noqa
                        np.cos(long2_rad-long1_rad))
    diff_sq = (angles_jit_deg - angles_np)**2
    assert diff_sq.sum() < 1e-10


def test_angle_jit_rad():
    size = 20000
    lat1 = np.random.uniform(low=-90, high=90, size=size)
    lat2 = np.random.uniform(low=-90, high=90, size=size)
    long1 = np.random.uniform(low=0, high=360, size=size)
    long2 = np.random.uniform(low=0, high=360, size=size)

    lat1_rad = np.deg2rad(lat1)
    lat2_rad = np.deg2rad(lat2)
    long1_rad = np.deg2rad(long1)
    long2_rad = np.deg2rad(long2)

    result_angles_deg = np.array(
        map(angle_jit_deg, lat1, long1, lat2, long2))
    result_angles_rad = np.array(
        map(pairwiser.angle_jit_rad, lat1_rad, long1_rad, lat2_rad,
            long2_rad))
    diff_sq = (result_angles_deg - result_angles_rad)**2
    assert diff_sq.sum() < 1e-10


def test_make_rsep():
    rsep = pairwiser.make_rsep(3, 1.0)
    diff_sq = np.array([0.5, 1.5, 2.5]) - rsep
    assert diff_sq.sum() < 1e-10


@numba.jit(nopython=True)  #todo change this to accept angles in rad
def angle_jit_deg(lat1, lon1, lat2, lon2):
    '''get angular distance between two points on a sphere
    (takes decimal degrees,return radians)'''
    lat1r = mt.radians(lat1)
    lon1r = mt.radians(lon1)
    lat2r = mt.radians(lat2)
    lon2r = mt.radians(lon2)
    ang = mt.acos(mt.sin(lat1r)*mt.sin(lat2r) +  # noqa
                  mt.cos(lat1r)*mt.cos(lat2r)*mt.cos(lon2r-lon1r))
    return ang


def test_pairwiser_ksz():
    length = 10000
    Dc = np.random.uniform(low=100, high=110, size=length)
    ra_rad = np.random.uniform(low=0, high=2*np.pi, size=length)
    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.random.uniform(low=-30, high=30, size=length)
    tzav = np.zeros(length)
    Tmapsc = np.random.uniform(low=0, high=20, size=length)
    nrbin = 40
    binsz = 5

    rsep, pest = pairwiser.pairwise_ksz(Dc, ra_deg, dec_deg, tzav,
                                        Tmapsc, binsz, nrbin,
                                        multithreading=False)

    rsep_v2, pest_v2 = pairwise_kSZ_fromV2(Dc, ra_deg, dec_deg, tzav, Tmapsc,
                                           binsz, nrbin=nrbin,
                                           multithreading=False)
    diff_sq = (pest - pest_v2)**2
    assert np.sum(diff_sq) < 1e-10


def test_pairwiser_one_row():
    length = 10000
    row = np.random.randint(0, length)
    Dc = np.random.uniform(low=100, high=110, size=length)
    ra_rad = np.random.uniform(low=0, high=2*np.pi, size=length)
    ra_deg = np.rad2deg(ra_rad)
    dec_rad = np.random.uniform(low=-30, high=30, size=length)
    dec_deg = np.rad2deg(dec_rad)
    tzav = np.zeros(length)
    Tmapsc = np.random.uniform(low=0, high=20, size=length)
    nrbin = 40
    binsz = 5
    dTw_pairwise_one_row = np.zeros(nrbin)
    w2_pairwise_one_row = np.zeros(nrbin)
    dTw_pariwise_one_rowV2 = np.zeros(nrbin)
    w2_pairwise_one_rowV2 = np.zeros(nrbin)
    pairwiser.pairwise_one_row(row, Dc, ra_rad, dec_rad, tzav,
                               Tmapsc, nrbin, binsz, dTw_pairwise_one_row,
                               w2_pairwise_one_row)
    pairwise_one_row_FromV2(row, Dc, ra_deg, dec_deg, length, tzav,
                            Tmapsc, binsz, dTw_pariwise_one_rowV2,
                            w2_pairwise_one_rowV2, nrbin=nrbin)
    sum_err_sq1 = np.sum((dTw_pairwise_one_row - dTw_pariwise_one_rowV2)**2)
    sum_err_sq2 = np.sum((w2_pairwise_one_row - w2_pairwise_one_rowV2)**2)
    assert sum_err_sq1 < 1e-10 and sum_err_sq2 < 1e-10


@numba.jit(nopython=True, nogil=True)
def pairwise_one_row_FromV2(row, Dc, ra, dec, many, tzav,
                            Tmapsc, binsz, dTw, w2, nrbin=40):
    '''This needs dTw and w2 to be numpy arrays of length nrbin.
    row, Dc, ra, dec, many tzav, Tmapsc, and binsz are defined as usual.
    Take a look at pairwise_it2 for definitions.'''
    i = row  #same notation as in pairwise_it
    for j in range(i+1, many):
        ang_ij = angle_jit_deg(dec[i], ra[i], dec[j], ra[j])
        vecdiff_ij = pairwiser.vecdiff_jit(Dc[i], Dc[j], ang_ij)
        binval_ij = int(vecdiff_ij/binsz)

        if binval_ij < nrbin:  #fits in ws and dTw
            dT_ij = (Tmapsc[i] - tzav[i])-(Tmapsc[j]-tzav[j])
            cij = (Dc[i]-Dc[j])*(1.0 + mt.cos(ang_ij)) / (2*vecdiff_ij)
            dTw[binval_ij] += dT_ij * cij
            w2[binval_ij] += cij**2.


def pairwise_kSZ_fromV2(Dc, ra, dec, tzav, Tmapsc, binsz,
                        nrbin=40, multithreading=True,
                        Nthreads=numba.config.NUMBA_NUM_THREADS):
    '''Iterates over the entire array computing all the pairs in Hand et al.
    numpy arrays:   Dc, ra, dec, tzav, Tmapsc.
    Scalars:        binsz, nrbin

    Arguments are:
                    Dc: distances
                    ra, dec: right ascention, decs in degrees
                    tzav: corrected temperatures
                    Tmapsc: Tmaps - cmb

                    binsz: how big is one bin in Mpc
                    nrbin: number of bins
    '''
#list of iterators for pairwise_one_row
#pylint: disable=unused-variable
    assert len(Dc) == len(ra) and len(ra) == len(dec)
    assert len(tzav) == len(dec) and len(tzav) == len(Tmapsc)
    many = len(ra)

    dTws = [np.zeros(nrbin) for j in range(many-1)]  #this is important!
    w2s = [np.zeros(nrbin) for j in range(many-1)]

    rows = xrange(many-1)
    Dcs = itertools.repeat(Dc, many-1)
    ras = itertools.repeat(ra, many-1)
    decs = itertools.repeat(dec, many-1)
    manys = itertools.repeat(many, many-1)
    tzavs = itertools.repeat(tzav, many-1)
    Tmapscs = itertools.repeat(Tmapsc, many-1)
    binszs = itertools.repeat(binsz, many-1)

    if multithreading:
        print "Running in %i threads..." % Nthreads
        with futures.ThreadPoolExecutor(Nthreads) as ex:
            ex.map(
                pairwise_one_row_FromV2,
                rows, Dcs,
                ras, decs, manys, tzavs,
                Tmapscs, binszs, dTws, w2s)

    else:
        print "Running on only one thread."
        map(
            pairwise_one_row_FromV2, rows, Dcs, ras, decs, manys, tzavs,
            Tmapscs, binszs, dTws, w2s)

    dTws = np.array(dTws)
    w2s = np.array(w2s)

    dTw = dTws.sum(axis=0)
    w2 = w2s.sum(axis=0)

    assert not np.any(w2 == 0)

    pest = -dTw/w2
    rsep = pairwiser.make_rsep(nrbin, binsz)

    return rsep, pest


def produceFakeCatalog():
    ''' Returns a fake pandas dataframe with data for pairwiser_ksz'''
    #produce fake data
    from iskay import cosmology
    import pandas as pd

    Nobj = 10000
    z = np.random.uniform(0, 1, Nobj)
    Dc = cosmology.Dc(z)
    ra_deg = np.random.uniform(0, 350, Nobj)
    dec_deg = np.random.uniform(-30, 0, Nobj)
    dT = np.random.uniform(-300, 300, Nobj)
    datain = {'z': z, 'Dc': Dc, 'ra': ra_deg, 'dec': dec_deg, 'dT': dT}
    df = pd.DataFrame(datain)
    return df
    #end produce fake data


def test_get_pairwise_ksz():
    testPath = '/'.join((catalogTools.__file__).split('/')[:-2]) + '/tests/'
    testParamFileFullPath = os.path.join(testPath, 'data_toTestAPI/params.ini')
    params = paramTools.params(testParamFileFullPath)

    df = produceFakeCatalog()
    rsep, p_uk = pairwiser.get_pairwise_ksz(df, params, multithreading=False)

    tzav = pairwiser.get_tzav(df.dT.values, df.z.values, params.SIGMA_Z)

    if not params.UNEVEN_BINS:
        rsep0, p_uk0 = pairwiser.pairwise_ksz(df.Dc.values, df.ra.values,
                                              df.dec.values, tzav,
                                              df.dT.values,
                                              params.BIN_SIZE_MPC,
                                              params.N_BINS,
                                              multithreading=False)
    else:
        rsep0, p_uk0 = pairwiser.pairwise_ksz_uneven_bins(df.Dc.values,
                                                          df.ra.values,
                                                          df.dec.values,
                                                          tzav,
                                                          df.dT.values,
                                                          params.BIN_EDGES,
                                                          multithreading=False)

    rsep_diff_sq = np.sum((rsep - rsep0)**2)
    p_uk_diff_sq = np.sum((p_uk - p_uk0)**2)
    assert rsep_diff_sq < 1e-10
    assert p_uk_diff_sq < 1e-10


def test_get_tzav_fast():
    N_gals = 15000
    sigma_z = 0.01
    z = np.random.uniform(size=N_gals)
    dT = np.random.normal(size=N_gals)

    tzav = pairwiser.get_tzav(dT, z, sigma_z)
    tzav_fast = pairwiser.get_tzav_fast(dT, z, sigma_z)

    chisq = np.sum((tzav-tzav_fast)**2)
    assert chisq < 1e-8


def test_varianceWeighted():
    '''Tests variance_weighted_pairwise_ksz and
    variance_weighted_pairwise_one_row'''
    testPath = '/'.join((catalogTools.__file__).split('/')[:-2]) + '/tests/'
    testParamFileFullPath = os.path.join(testPath, 'data_toTestAPI/params.ini')
    params = paramTools.params(testParamFileFullPath)

    df = produceFakeCatalog()
    rsep, p_uk = pairwiser.get_pairwise_ksz(df, params, multithreading=False)

    tzav = pairwiser.get_tzav(df.dT.values, df.z.values, params.SIGMA_Z)
    div = np.ones(len(tzav))
    rsep0, p_uk0 = pairwiser.variance_weighted_pairwise_ksz(df.Dc.values,
                                          df.ra.values,  # noqa
                                          df.dec.values, tzav, df.dT.values,
                                          div,
                                          params.BIN_SIZE_MPC, params.N_BINS,
                                          multithreading=False)
    chisq = np.sum((p_uk - p_uk0)**2)
    assert chisq < 1.0e10


def test_make_rsep_uneven_bins():
    bin_edges = np.array([5, 15, 25, 35, 45, 70, 90])
    bins = pairwiser.make_rsep_uneven_bins(bin_edges)
    bins_tocompare = np.array([10, 20, 30, 40, 57.5, 80])
    chisq = np.sum((bins-bins_tocompare)**2)
    assert chisq < 1e-10


def test_pairwiser_ksz_uneven_bins():
    length = 10000
    Dc = np.random.uniform(low=100, high=110, size=length)
    ra_rad = np.random.uniform(low=0, high=2*np.pi, size=length)
    ra_deg = np.rad2deg(ra_rad)
    dec_deg = np.random.uniform(low=-30, high=30, size=length)
    tzav = np.zeros(length)
    Tmapsc = np.random.uniform(low=0, high=20, size=length)
    nrbin = 40
    binsz = 5
    bin_edges = np.arange(0, binsz*(nrbin+1.0), binsz)

    rsep, pest = pairwiser.pairwise_ksz(Dc, ra_deg, dec_deg, tzav,
                                        Tmapsc, binsz, nrbin,
                                        multithreading=False)

    rsep_uneven, pest_uneven = pairwiser.pairwise_ksz_uneven_bins(Dc, ra_deg,
                                                    dec_deg,  # noqa
                                                   tzav, Tmapsc, # noqa
                                                   bin_edges, # noqa
                                                   multithreading=False) # noqa
    diff_sq = (pest - pest_uneven)**2
    assert np.sum(diff_sq) < 1e-10
    diff_sq = (rsep-rsep_uneven)**2
    assert np.sum(diff_sq) < 1e-10

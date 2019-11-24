'''This module contains the core pairwise kSZ estimator algorithm.
'''
import numba
import math as mt
import numpy as np
import itertools
from concurrent import futures
from scipy import interpolate


# get tzav
@numba.guvectorize(['float64[:],float64[:],float64[:],float64[:],'
                    'float64[:],float64[:]'],
                   '(n),(n),(),()->(),()', target='parallel')
def get_tzav_and_w_nb(dT, z, zj, sigma_z, res1, res2):
    '''Launched by get_tzav to compute formula in parallel '''
    for i in range(dT.shape[0]):
        res1 += dT[i] * mt.exp(-(zj[0]-z[i])**2.0/(2.0*sigma_z[0]**2))
        res2 += mt.exp(-(zj[0]-z[i])**2/(2.0*sigma_z[0]**2))


def get_tzav(dTs, zs, sigma_z):
    '''Computes the dT dependency to redshift.
    dTs: Temperature decrements
    zs: redshifts
    sigma_z: width of the gaussian to smooth out the moving window.'''
    #To test, run test_get_tzav_nb
    #   Create empty arrays to be used by numba in get_tzav_and_w_nb'''
    res1 = np.zeros(dTs.shape[0])
    res2 = np.zeros(dTs.shape[0])
    get_tzav_and_w_nb(dTs, zs, zs, sigma_z, res1, res2)
    return res1/res2  # tzav/w
#end get_tzav formulas


def get_tzav_fast(dTs, zs, sigma_z):
    '''Subsample and interpolate Tzav to make it fast.
    dTs: entire list of dT decrements
    zs: entire list of redshifts
    sigma_z: width of the gaussian kernel we want to apply.
    '''
    N_samples_in_sigmaz = 15  # in one width of sigmaz use Nsamples
    zmin, zmax = zs.min(), zs.max()
    delta_z = zmax - zmin

    # evaluates Tzav N times
    N_samples = int(round(delta_z/sigma_z)) * N_samples_in_sigmaz
    z_subsampled = np.linspace(zmin, zmax, N_samples)

    #now compute tzav as we usually do.
    res1 = np.zeros(z_subsampled.shape[0])
    res2 = np.zeros(z_subsampled.shape[0])
    get_tzav_and_w_nb(dTs, zs, z_subsampled, sigma_z, res1, res2)
    tzav_subsampled = res1/res2
    #interpolate
    f = interpolate.interp1d(z_subsampled, tzav_subsampled, kind='cubic')
    tzav_fast = f(zs)
    return tzav_fast


@numba.jit(nopython=True, nogil=True)
def pairwise_one_row(row, Dc, ra_rad, dec_rad, tzav,
                     Tmapsc, nrbin, binsz, dTw, w2):
    '''This needs dTw and w2 to be numpy arrays of length nrbin.
        row: what row to compute
        Dc: distance
        ra_rad: ra in rad
        dec_rad: dec in rad
        tzav: Average T as a function of redshift.
        Tmapsc: Tdisc - Tring_cmb
        nrbin: number of separation binszs for the kSZ estimator
        binsz: bin size in Mpc
        dTw: numpy array of size nrbin
        w2: idem'''
    many = len(ra_rad)
    i = row  #same notation as in pairwise_it
    for j in range(i+1, many):
        ang_ij = angle_jit_rad(dec_rad[i], ra_rad[i], dec_rad[j], ra_rad[j])
        vecdiff_ij = vecdiff_jit(Dc[i], Dc[j], ang_ij)
        binval_ij = int(vecdiff_ij/binsz)

        if binval_ij < nrbin:  #fits in ws and dTw
            dT_ij = (Tmapsc[i] - tzav[i])-(Tmapsc[j]-tzav[j])
            cij = (Dc[i]-Dc[j])*(1.0 + mt.cos(ang_ij)) / (2*vecdiff_ij)
            dTw[binval_ij] += dT_ij * cij
            w2[binval_ij] += cij**2.


def pairwise_ksz(Dc, ra_deg, dec_deg, tzav, Tmapsc,
                 binsz, nrbin,
                 multithreading=True, Nthreads=numba.config.NUMBA_NUM_THREADS):
    '''Produces the ksz curve for givne arguments.
        Dc: radial distance
        ra_deg, dec_deg: ra and dec of the object in degrees
        tazv: smoothed out redshift dependent temperature
        Tmapsc: temperature decrement
        binsz, nrbin: bin size in Mpc and number of separation bins
        multithreading sets if you want to run in different threads for
        Nthreads or if you want a gigantic for loop.
    '''
    assert len(Dc) == len(ra_deg) == len(dec_deg) == len(tzav) == len(Tmapsc)
    length = len(ra_deg)

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    dTws = [np.zeros(nrbin) for j in range(length-1)]  #list of results for
    w2s = [np.zeros(nrbin) for j in range(length-1)]  #each pairwiser row

    rows = xrange(length-1)  # iterate over (n-1) rows for i!=j ...
    Dcs = itertools.repeat(Dc, length-1)
    ras_rad = itertools.repeat(ra_rad, length-1)
    decs_rad = itertools.repeat(dec_rad, length-1)
    tzavs = itertools.repeat(tzav, length-1)
    Tmapscs = itertools.repeat(Tmapsc, length-1)
    nrbins = itertools.repeat(nrbin, length-1)
    binszs = itertools.repeat(binsz, length-1)

    if multithreading:
        print "Running in %i threads..." % Nthreads
        with futures.ThreadPoolExecutor(Nthreads) as ex:
            ex.map(
                pairwise_one_row,
                rows, Dcs, ras_rad, decs_rad, tzavs, Tmapscs,
                nrbins, binszs, dTws, w2s)
    else:
        print "Running on only one thread."
        map(
            pairwise_one_row,
            rows, Dcs, ras_rad, decs_rad, tzavs, Tmapscs,
            nrbins, binszs, dTws, w2s)

    dTws = np.array(dTws)
    w2s = np.array(w2s)

    dTw = dTws.sum(axis=0)
    w2 = w2s.sum(axis=0)

    assert not np.any(w2 == 0)

    pest = -dTw/w2
    rsep = make_rsep(nrbin, binsz)

    return rsep, pest


@numba.jit(nopython=True)
def angle_jit_rad(lat1r, lon1r, lat2r, lon2r):
    '''Gets angular distance between two points on a sphere.
    All numbers must be in radians, returns radians.'''
    ang = mt.acos(mt.sin(lat1r)*mt.sin(lat2r) +  # noqa
                  mt.cos(lat1r)*mt.cos(lat2r)*mt.cos(lon2r-lon1r))
    return ang


@numba.jit(nopython=True)
def vecdiff_jit(d1, d2, angle_rad):
    ''' module of the difference vector between d2 and d1,
        where a is the angle between them'''
    r = mt.sqrt(d2**2. + d1**2. - 2.*d2*d1*mt.cos(angle_rad))
    return r


def make_rsep(nrbin, binsz):
    '''Generates the x axis of the histogram.
    Bin positions are halfway of the step.
    In the following diagram, "x" marks the rsep histogram axis.
    the "|" mark the edge of each bin.

    Notice that if a point falls between 0 and 1 int(r_sep/binsz) it
    is asigned to the first bin, and labeled rsep_bin = 0.5
    If a point falls between 1 and 2 it is labeled rsep_bin = 1.5 and so
    forth.
    |  x  |  x  |  x  |  x  |
    0     1     2     3     4  -> rsep
      0.5   1.5   2.5   3.5    -> rsep_bins
    Arguments are:
        binsz: bin size
        nrbin: number of bins
    '''
    return np.linspace(0, (nrbin-1) * binsz, nrbin) + binsz/2.


def get_pairwise_ksz(df, params,
                     multithreading=False):
    '''Wrapper for calling pairwise_ksz using only a preprocessed dataframe
    and a parameters container.
    Arguments:
        df: preprocessed catalog
        params: paramTols.params object.'''
    dT = df.dT.values
    z = df.z.values
    Dc = df.Dc.values

    #can add options for alternative evaluations of tzav
    if params.GET_TZAV_FAST:
        tzav = get_tzav_fast(dT, z, params.SIGMA_Z)  # use fast implementation
    else:
        tzav = get_tzav(dT, z, params.SIGMA_Z)  # use slow one

    ra_deg = df.ra.values
    dec_deg = df.dec.values
    binsz = params.BIN_SIZE_MPC
    nbins = params.N_BINS

    if params.DO_VARIANCE_WEIGHTED:
        print("need to implement this!")
        div = df.div_disk.values
        r_sep, p_uk = variance_weighted_pairwise_ksz(Dc, ra_deg, dec_deg,
                                                     tzav, dT, div,
                                                     binsz, nbins,
                                    multithreading=multithreading)  # noqa
    else:
        r_sep, p_uk = pairwise_ksz(Dc, ra_deg, dec_deg, tzav, dT,
                                   binsz, nbins,
                                   multithreading=multithreading)
    return r_sep, p_uk

#variance weighted implementation. These funcitons are basically lcones of
#pairwise_ksz and pairwise one row.


def variance_weighted_pairwise_ksz(Dc, ra_deg, dec_deg, tzav, Tmapsc, div,
                                   binsz, nrbin,
                                   multithreading=True,
                                   Nthreads=numba.config.NUMBA_NUM_THREADS):
    '''Produces the ksz curve for givne arguments.
        Dc: radial distance
        ra_deg, dec_deg: ra and dec of the object in degrees
        tazv: smoothed out redshift dependent temperature
        Tmapsc: temperature decrement
        divs: inverse variance weights
        binsz, nrbin: bin size in Mpc and number of separation bins
        multithreading sets if you want to run in different threads for
        Nthreads or if you want a gigantic for loop.
    '''
    assert len(Dc) == len(ra_deg) == len(dec_deg) == len(tzav) == len(Tmapsc)
    length = len(ra_deg)
    assert len(div) == length

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    dTws = [np.zeros(nrbin) for j in range(length-1)]  #list of results for
    w2s = [np.zeros(nrbin) for j in range(length-1)]  #each pairwiser row

    rows = xrange(length-1)  # iterate over (n-1) rows for i!=j ...
    Dcs = itertools.repeat(Dc, length-1)
    ras_rad = itertools.repeat(ra_rad, length-1)
    decs_rad = itertools.repeat(dec_rad, length-1)
    tzavs = itertools.repeat(tzav, length-1)
    Tmapscs = itertools.repeat(Tmapsc, length-1)
    nrbins = itertools.repeat(nrbin, length-1)
    binszs = itertools.repeat(binsz, length-1)
    divs = itertools.repeat(div, length-1)

    if multithreading:
        print "Running in %i threads..." % Nthreads
        with futures.ThreadPoolExecutor(Nthreads) as ex:
            ex.map(variance_weighted_pairwise_one_row,
                   rows, Dcs, ras_rad, decs_rad, tzavs, Tmapscs, divs,
                   nrbins, binszs, dTws, w2s)
    else:
        print "Running on only one thread."
        map(variance_weighted_pairwise_one_row,
            rows, Dcs, ras_rad, decs_rad, tzavs, Tmapscs, divs,
            nrbins, binszs, dTws, w2s)

    dTws = np.array(dTws)
    w2s = np.array(w2s)

    dTw = dTws.sum(axis=0)
    w2 = w2s.sum(axis=0)

    assert not np.any(w2 == 0)

    pest = -dTw/w2
    rsep = make_rsep(nrbin, binsz)

    return rsep, pest


@numba.jit(nopython=True, nogil=True)
def variance_weighted_pairwise_one_row(row, Dc, ra_rad, dec_rad, tzav,
                                       Tmapsc, divs, nrbin, binsz, dTw, w2):
    '''This needs dTw and w2 to be numpy arrays of length nrbin.
        row: what row to compute
        Dc: distance
        ra_rad: ra in rad
        dec_rad: dec in rad
        tzav: Average T as a function of redshift.
        divs: inverse variances from the map
        Tmapsc: Tdisc - Tring_cmb
        nrbin: number of separation binszs for the kSZ estimator
        binsz: bin size in Mpc
        dTw: numpy array of size nrbin
        w2: idem'''
    many = len(ra_rad)
    i = row  #same notation as in pairwise_it
    for j in range(i+1, many):
        ang_ij = angle_jit_rad(dec_rad[i], ra_rad[i], dec_rad[j], ra_rad[j])
        vecdiff_ij = vecdiff_jit(Dc[i], Dc[j], ang_ij)
        binval_ij = int(vecdiff_ij/binsz)

        if binval_ij < nrbin:  #fits in ws and dTw
            dT_ij = (Tmapsc[i] - tzav[i])-(Tmapsc[j]-tzav[j])
            cij = (Dc[i]-Dc[j])*(1.0 + mt.cos(ang_ij)) / (2*vecdiff_ij)
            sigma_sq = 1.0/divs[i] + 1.0/divs[j]
            dTw[binval_ij] += dT_ij * cij/sigma_sq
            w2[binval_ij] += cij**2./sigma_sq

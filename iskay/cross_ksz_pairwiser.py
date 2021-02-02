'''
Core engine for pairwise cross survey estimator.

Originally written to compute covariance matrices of 90x150 maps.

By P. Gallardo.
'''
import numpy as np
import itertools
from iskay.pairwiser import get_tzav
from iskay.pairwiser import get_tzav_fast
from iskay.pairwiser import angle_jit_rad
from iskay.pairwiser import vecdiff_jit
from iskay.pairwiser import inWhatBinIsIt
from iskay.pairwiser import make_rsep_uneven_bins
import math as mt
import numba
from concurrent import futures


def get_cross_pairwise_ksz(df1, df2, params):
    '''
    Wrapper for calling pairwise_cross_ksz using a cross_dataset
    object.

    params: params object from paramTools.py
    '''
    dT1 = df1.dT.values
    dT2 = df2.dT.values
    z = df1.z.values  # this requires Dc and z values to be the same
    Dc = df1.Dc.values

    if params.GET_TZAV_FAST:
        tzav1 = get_tzav_fast(dT1, z, params.SIGMA_Z)
        tzav2 = get_tzav_fast(dT2, z, params.SIGMA_Z)
    else:
        tzav1 = get_tzav(dT1, z, params.SIGMA_Z)
        tzav2 = get_tzav(dT2, z, params.SIGMA_Z)

    # more common values
    ra_deg = df1.ra.values
    dec_deg = df1.dec.values

    assert params.UNEVEN_BINS  # only uneven bins supported for this
    bin_edges = params.BIN_EDGES

    if params.DO_VARIANCE_WEIGHTED:
        assert False  # not implemented
    else:
        r_sep, p_uk = cross_pairwise_ksz_uneven_bins(Dc, ra_deg, dec_deg,
                                                     tzav1, tzav2,
                                                     dT1, dT2,
                                                     bin_edges)
    return r_sep, p_uk


def cross_pairwise_ksz_uneven_bins(Dc, ra_deg, dec_deg, tzav1, tzav2,
                                   dT1, dT2, bin_edges,
                                   Nthreads=numba.config.NUMBA_NUM_THREADS):
    ''' Produces ksz curve across two instruments.
    Dc, ra and dec are shared (must be the same across both instruments)
    tazv, dT come from both experiments.
    bin_edges must be a vector of bin edges.
    '''
    nrbin = len(bin_edges) - 1
    assert len(Dc) == len(ra_deg) == len(dT1) == len(dT2) == len(tzav1)

    length = len(ra_deg)

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    dTws = [np.zeros(nrbin) for j in range(length-1)]
    w2s = [np.zeros(nrbin) for j in range(length-1)]

    rows = xrange(length-1)
    Dcs = itertools.repeat(Dc, length-1)
    ras_rad = itertools.repeat(ra_rad, length-1)
    decs_rad = itertools.repeat(dec_rad, length-1)
    tzav1s = itertools.repeat(tzav1, length-1)
    tzav2s = itertools.repeat(tzav2, length-1)
    dT1s = itertools.repeat(dT1, length-1)
    dT2s = itertools.repeat(dT2, length-1)
    bin_edges_s = itertools.repeat(bin_edges, length-1)

    with futures.ThreadPoolExecutor(Nthreads) as ex:
        ex.map(cross_pairwiser_one_row_uneven_bins,
               rows, Dcs, ras_rad, decs_rad, tzav1s, tzav2s, dT1s, dT2s,
               bin_edges_s, dTws, w2s)

    dTws = np.array(dTws)
    w2s = np.array(w2s)

    dTw = dTws.sum(axis=0)
    w2 = w2s.sum(axis=0)

    assert not np.any(w2 == 0)

    pest = -dTw/w2
    rsep = make_rsep_uneven_bins(bin_edges)

    return rsep, pest


@numba.jit(nopython=True, nogil=True)
def cross_pairwiser_one_row_uneven_bins(row, Dc, ra_rad, dec_rad, tzav1, tzav2,
                                        dT1, dT2, bin_edges, dTw, w2):
    '''Idem to pairwise_one_row_uneven_bins from pairwiser.py'''
    many = len(ra_rad)
    i = row
    for j in range(i+1, many):
        ang_ij = angle_jit_rad(dec_rad[i], ra_rad[i], dec_rad[j], ra_rad[j])
        vecdiff_ij = vecdiff_jit(Dc[i], Dc[j], ang_ij)
        #  check if separation within bin space
        if (vecdiff_ij > bin_edges[0]) and (vecdiff_ij < bin_edges[-1]):
            binval_ij = inWhatBinIsIt(vecdiff_ij, bin_edges)
            dT_ij = (dT1[i]-tzav1[i]) - (dT2[j]-tzav2[j])
            cij = (Dc[i] - Dc[j]) * (1.0 + mt.cos(ang_ij))/(2.0*vecdiff_ij)
            dTw[binval_ij] += dT_ij * cij
            w2[binval_ij] += cij**2.0

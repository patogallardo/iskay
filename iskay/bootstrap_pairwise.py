from iskay import pairwiser
import numpy as np
from scipy import interpolate
import progressbar
from iskay.pairwiser import angle_jit_rad
from iskay.pairwiser import vecdiff_jit
from iskay.pairwiser import inWhatBinIsIt
from iskay.pairwiser import make_rsep_uneven_bins
import numba
import math as mt
import itertools
from concurrent import futures


def get_tzav_on_few_points(dTs, zs, sigma_z, z_subsampled):
    '''Subsample the average in z space and generate a model
    f(z) for th ebootstrap.
    '''
    zmin, zmax = zs.min(), zs.max()
    N_samples = len(z_subsampled)
    z_subsampled = np.linspace(zmin, zmax, N_samples)

    res1 = np.zeros(z_subsampled.shape[0])
    res2 = np.zeros(z_subsampled.shape[0])
    pairwiser.get_tzav_and_w_nb(dTs, zs, z_subsampled, sigma_z, res1, res2)
    tzav_subsampled = res1/res2

    f = interpolate.interp1d(z_subsampled, tzav_subsampled, kind='cubic',
                             fill_value='extrapolate')
    return f


def bootstrap_wigglyTee(df, params, N_boot, N_samples_in_sigmaz):
    tzavs = []
    delta_z = df.z.values.max() - df.z.values.min()
    N_samples = int(round(delta_z/params.SIGMA_Z)) * N_samples_in_sigmaz
    z_subsampled = np.linspace(df.z.min(), df.z.max(), N_samples)

    f_full_dataset = get_tzav_on_few_points(df.dT.values, df.z.values,
                                            params.SIGMA_Z, z_subsampled)

    for iteration in progressbar.progressbar(range(N_boot)):
        df_sample = df.sample(len(df), replace=True)
        dTs = df_sample.dT.values
        zs = df_sample.z.values
        f = get_tzav_on_few_points(dTs, zs, params.SIGMA_Z, z_subsampled)
        tzavs.append(f(z_subsampled))

    tzavs = np.array(tzavs)
    sigmas = np.std(tzavs, axis=0)
    sigma_model = interpolate.interp1d(z_subsampled, sigmas, kind='cubic',
                                       fill_value='extrapolate')
    mean_model = f_full_dataset
    sigma_model = sigma_model
    return z_subsampled, mean_model, sigma_model


class modelWigglyTee():
    def __init__(self, df, params, N_boot, N_samples_in_sigmaz=20):
        self.df = df
        self.params = params
        self.N_boot = N_boot
        self.N_samples_in_sigmaz = N_samples_in_sigmaz

        z_subsampled, mean_model, sigma_model = bootstrap_wigglyTee(df,
                params, N_boot, N_samples_in_sigmaz)  # noqa
        self.z_subsampled = z_subsampled
        self.mean_model = mean_model
        self.sigma_model = sigma_model

    def tzav_with_noise_model(self):
        tzav_noiseless = self.mean_model(self.df.z.values)
        sigmas = self.sigma_model(self.df.z.values)
        tzav_noisy = np.random.normal(loc=tzav_noiseless, scale=sigmas)
        return tzav_noisy


def howManyPairsInUpperTriangle(N_gal):
    '''returns the number of upper triangle elements there
    are in a matrix of side N.'''
    return int((N_gal-1)*N_gal/2)


@numba.jit(nopython=True)
def gen_random_pair(N, out):
    '''Generates a random pair in the upper
        triangle of a matrix.
        N: is the length of objects to generate pairs.
        out is a numpy array of length=2
        out = i, j'''
    onceMore = True
    while onceMore:
        i = np.random.randint(0, N)
        j = np.random.randint(0, N)
        onceMore = False  # by default end here
        if j < i:  # if lower triangle, swap them
            swap = i
            i = j
            j = swap
        elif i == j:  # if diagonal, try again
            onceMore = True
        out[0] = i
        out[1] = j


@numba.jit(nopython=True, nogil=True)
def pairwise_Nit_uneven_bins(Nit, Dc, ra_rad, dec_rad, tzav,
                             Tmapsc, bin_edges, dTw, w2):
    N_gal = len(Dc)
    rand_ij = np.zeros(2, dtype=np.int64)
    for it in xrange(Nit):
        gen_random_pair(N_gal, rand_ij)
        i, j = rand_ij[0], rand_ij[1]
        ang_ij = angle_jit_rad(dec_rad[i], ra_rad[i], dec_rad[j], ra_rad[j])
        vecdiff_ij = vecdiff_jit(Dc[i], Dc[j], ang_ij)
        #  Check if this separation is within the bin space
        if (vecdiff_ij > bin_edges[0]) and (vecdiff_ij < bin_edges[-1]):
            binval_ij = inWhatBinIsIt(vecdiff_ij, bin_edges)
        #  Now compute and store
            dT_ij = (Tmapsc[i] - tzav[i])-(Tmapsc[j]-tzav[j])
            cij = (Dc[i]-Dc[j])*(1.0 + mt.cos(ang_ij)) / (2.0*vecdiff_ij)
            dTw[binval_ij] += dT_ij * cij
            w2[binval_ij] += cij**2.


def partition_pw_computation(N_gal, chunk_length):
    '''Receives the number of galaxies and the
    number of pairs we are willing to compute for each subprocess.
    Returns how many times we need to compute pairs of length N_chunk and
    the number of iterations needed in the last chunk.

    returns N_chunks, last_chunk_length
    '''
    N_pairs = howManyPairsInUpperTriangle(N_gal)
    N_equal_chunks = N_pairs//chunk_length
    last_chunk_length = N_pairs % chunk_length
    return N_equal_chunks, last_chunk_length


def bootstrapped_pairwise_ksz_uneven_bins(Dc, ra_deg, dec_deg, tzav, Tmapsc,
                                          bin_edges,
                                          do_variance_weighted, divs=None,
                                          multithreading=True,
                            Nthreads=numba.config.NUMBA_NUM_THREADS,
                            chunk_length=100000000):  # noqa
    nrbin = len(bin_edges) - 1
    assert len(Dc) == len(ra_deg) == len(dec_deg) == len(tzav) == len(Tmapsc)
    N_gal = len(ra_deg)

    ra_rad = np.deg2rad(ra_deg)
    dec_rad = np.deg2rad(dec_deg)

    N_equal_chunks, last_chunk_length = partition_pw_computation(N_gal,
                                                                 chunk_length)

    if last_chunk_length > 0:
        N_iterations = N_equal_chunks + 1
        chunk_lengths = [chunk_length] * N_equal_chunks + [last_chunk_length]

    elif last_chunk_length == 0:
        N_iterations = N_equal_chunks
        chunk_lengths = [chunk_lengths] * N_equal_chunks

    dTws = [np.zeros(nrbin) for j in range(N_iterations)]  #list of results for
    w2s = [np.zeros(nrbin) for j in range(N_iterations)]  #each pairwiser iter

    Dcs = itertools.repeat(Dc, N_iterations)
    ras_rad = itertools.repeat(ra_rad, N_iterations)
    decs_rad = itertools.repeat(dec_rad, N_iterations)
    tzavs = itertools.repeat(tzav, N_iterations)
    Tmapscs = itertools.repeat(Tmapsc, N_iterations)
    bin_edges_s = itertools.repeat(bin_edges, N_iterations)

    if do_variance_weighted:
        assert False  # not implemented
        assert len(divs) == len(Dc)
        divs_s = itertools.repeat(divs, N_iterations)  # noqa

    if multithreading:
        print("Running in %i threads." % Nthreads)
        if do_variance_weighted:
            assert False  # not implemented
        else:
            with futures.ThreadPoolExecutor(Nthreads) as ex:
                ex.map(
                    pairwise_Nit_uneven_bins,
                    chunk_lengths, Dcs, ras_rad, decs_rad, tzavs,
                    Tmapscs, bin_edges_s, dTws, w2s)
    else:
        print("Running on only one thread")
        if do_variance_weighted:
            assert False  # not implemented
        else:
            map(pairwise_Nit_uneven_bins,
                chunk_lengths, Dcs, ras_rad, decs_rad, tzavs,
                Tmapscs, bin_edges_s, dTws, w2s)

    dTws = np.array(dTws)
    w2s = np.array(w2s)

    dTw = dTws.sum(axis=0)
    w2 = w2s.sum(axis=0)

    assert not np.any(w2 == 0)

    pest = -dTw/w2
    rsep = make_rsep_uneven_bins(bin_edges)

    return rsep, pest


def get_bootstrap_pairwise(df, params, multithreading=False):
    '''
    Wrapper for bootstrapping pairs in the pw estimator.
    It requires one column to have a tzav (which can be zero).
    This is because if you want to resample in pair space, you need to
    add a noise model for wiggly tee once for all the replicant samples.
    '''
    dT = df.dT.values
    Dc = df.Dc.values
    tzav = df.tzav.values  # requires it to exsist in the dataframe
    ra_deg = df.ra.values
    dec_deg = df.dec.values

    bin_edges = params.BIN_EDGES

    if params.DO_VARIANCE_WEIGHTED:
        assert False  # Not implemented
    else:
        r_sep, p_uk = bootstrapped_pairwise_ksz_uneven_bins(Dc, ra_deg,
                            dec_deg, tzav, dT, # noqa
                            bin_edges,
                            do_variance_weighted=False,
                            divs=None,
                            multithreading=multithreading,
                            chunk_length=100000000)
    return r_sep, p_uk

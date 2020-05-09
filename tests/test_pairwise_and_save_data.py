from iskay import pairwise_and_save_data
from iskay import pairwiser
import numpy as np
from scipy import stats


def test_do_the_jitted_pairwise():
    Ngal = 10000
    row = np.random.randint(0, Ngal, 1)[0]
    rmax = 400
    ra_rad, dec_rad, Dc, dT_ksz = generate_data(Ngal)

    vecdiff_ij_s = np.zeros(Ngal)
    cij_s = np.zeros(Ngal)
    dTw_s = np.zeros(Ngal)
    w2_s = np.zeros(Ngal)
    i_s = np.zeros(Ngal)
    j_s = np.zeros(Ngal)

    nstored = pairwise_and_save_data.do_the_jitted_pairwise(
              ra_rad, dec_rad, Dc, dT_ksz,  # noqa
              row, rmax,  # noqa
              vecdiff_ij_s, cij_s, dTw_s, w2_s,  # noqa
              i_s, j_s)  # noqa
    vecdiff_ij_s = vecdiff_ij_s[:nstored]
    cij_s = cij_s[:nstored]
    dTw_s = dTw_s[:nstored]
    w2_s = w2_s[:nstored]
    i_s = i_s[:nstored]
    j_s = j_s[:nstored]

    #now do the dame with pairwise_one_row
    tzav = np.zeros(Ngal)
    nrbin = 40
    binsz = 10
    dTw_pw = np.zeros(nrbin)
    w2_pw = np.zeros(nrbin)

    pairwiser.pairwise_one_row(row, Dc, ra_rad,
                               dec_rad, tzav, dT_ksz,
                               nrbin, binsz, dTw_pw, w2_pw)

    bins = np.arange(0, (nrbin+1) * binsz, binsz)
    binned_dTws = stats.binned_statistic(vecdiff_ij_s,
                                         dTw_s,
                                         statistic='sum',
                                         bins=bins)[0]
    binned_w2 = stats.binned_statistic(vecdiff_ij_s,
                                       w2_s,
                                       statistic='sum',
                                       bins=bins)[0]

    assert np.sum((binned_w2 - w2_pw)**2) < 1e-10
    assert np.sum((binned_dTws - dTw_pw)**2) < 1e-10


def generate_data(length):
    ra_rad = np.random.uniform(low=0, high=np.pi,
                               size=length)
    dec_rad = np.random.uniform(low=-np.pi/6,
                                high=np.pi/6,
                                size=length)
    Dc = np.random.uniform(10, 200, size=length)
    dT_ksz = np.random.normal(0, 45, size=length)
    return ra_rad, dec_rad, Dc, dT_ksz

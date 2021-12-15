import numba
from pairwiser import angle_jit_rad, vecdiff_jit, inWhatBinIsIt
import math as mt


@numba.jit(nopython=True, nogil=True)  # row_end = roundup(N/2)
def pairwise_from_rowtorow(row_start, row_end,
                           Dc, ra_rad, dec_rad, tzav,
                           Tmapsc, bin_edges,
                           dTw, w2, w, dT, Npairs):
    Ngal = len(ra_rad)
    for row in range(row_start, row_end):
        pairwise_one_row(row, Ngal, Dc, ra_rad, dec_rad, tzav,
                         Tmapsc, bin_edges,
                         dTw, w2, w, dT, Npairs)
        mirror_row = Ngal - 1 - row
        if mirror_row > row:
            pairwise_one_row(mirror_row, Ngal, Dc, ra_rad, dec_rad, tzav,
                             Tmapsc, bin_edges,
                             dTw, w2, w, dT, Npairs)


@numba.jit(nopython=True, nogil=True)  # row_end = roundup(N/2)
def pairwise_from_rowtorow_onlyonerowatatime(row_start, row_end,
                                             Dc, ra_rad, dec_rad, tzav,
                                             Tmapsc, bin_edges,
                                             dTw, w2, w, dT, Npairs):
    Ngal = len(ra_rad)
    for row in range(row_start, row_end):
        pairwise_one_row(row, Ngal, Dc, ra_rad, dec_rad, tzav,
                         Tmapsc, bin_edges,
                         dTw, w2, w, dT, Npairs)


@numba.jit(nopython=True, nogil=True)
def pairwise_one_row(row, Ngal, Dc, ra_rad, dec_rad,
                     tzav,
                     Tmapsc, bin_edges,
                     dTw, w2, w, dT, Npairs):
    for j in range(row+1, Ngal):
        ang_ij = angle_jit_rad(dec_rad[row], ra_rad[row],
                               dec_rad[j], ra_rad[j])
        vecdiff_ij = vecdiff_jit(Dc[row], Dc[j], ang_ij)
        if (vecdiff_ij > bin_edges[0]) and (vecdiff_ij < bin_edges[-1]):
            binval_ij = inWhatBinIsIt(vecdiff_ij, bin_edges)
            dT_ij = (Tmapsc[row] - tzav[row]) - (Tmapsc[j] - tzav[j])
            cij = (Dc[row]-Dc[j])*(1.0 + mt.cos(ang_ij)) / (2.0*vecdiff_ij)
            dTw[binval_ij] += dT_ij * cij
            w2[binval_ij] += cij**2.
            dT[binval_ij] += dT_ij
            w[binval_ij] += cij
            Npairs[binval_ij] += 1

import numba
from iskay import pairwiser
import math as mt
import numpy as np
import pandas as pd
import os
from iskay.pairwiser import inWhatBinIsIt


@numba.jit(nopython=True)
def do_the_jitted_pairwise(ra_rad, dec_rad, Dc, dT_ksz,
                           row, bin_edges,
                           vecdiff_ij_s, cij_s, dTw_s, w2_s, i_s, j_s,
                           bin_number_s):
    '''Will do the pairwise calculation fast.
    returns n_stored which is the length of the vector that survived the
    r_max cut
    Results are stored in vecdiff_ij_s, cij_s, dTw_s, and w2_s.'''
    i = row
    N_gal = len(ra_rad)
    n_stored = 0
    for j in range(row+1, N_gal):
        ang_ij = pairwiser.angle_jit_rad(dec_rad[i], ra_rad[i],
                                         dec_rad[j], ra_rad[j])
        vecdiff_ij = pairwiser.vecdiff_jit(Dc[i], Dc[j], ang_ij)
        if (vecdiff_ij > bin_edges[0]) and (vecdiff_ij < bin_edges[-1]):
            binval_ij = inWhatBinIsIt(vecdiff_ij, bin_edges)
            cij = (Dc[i]-Dc[j])*(1.0 + mt.cos(ang_ij)) / (2*vecdiff_ij)
            dT_ij = dT_ksz[i]-dT_ksz[j]
            dTw = dT_ij * cij
            w2 = cij**2
            #store result
            vecdiff_ij_s[n_stored] = vecdiff_ij
            cij_s[n_stored] = cij
            dTw_s[n_stored] = dTw
            w2_s[n_stored] = w2
            i_s[n_stored] = row
            j_s[n_stored] = j
            bin_number_s[n_stored] = binval_ij
            n_stored += 1
    return n_stored


def pairwise_one_row_and_save_all_jitted(df, row, bin_edges,
                                         MAX_NPAIRS=1000000):
    '''This function sets up the jitted pairwise calculation.'''
    #declare variables that we will dneed
    ra_rad = df['ra_rad'].values
    dec_rad = df['dec_rad'].values
    Dc = df['Dc'].values
    dT_ksz = df['dT_kSZ'].values

    vecdiff_ij_s = np.zeros(MAX_NPAIRS)
    cij_s = np.zeros(MAX_NPAIRS)
    dTw_s = np.zeros(MAX_NPAIRS)
    w2_s = np.zeros(MAX_NPAIRS)
    i_s = np.zeros(MAX_NPAIRS, dtype=int)
    j_s = np.zeros(MAX_NPAIRS, dtype=int)
    bin_number_s = np.zeros(MAX_NPAIRS, dtype=int)

    #now call the pairwise calculation for this row
    n_pairs = do_the_jitted_pairwise(ra_rad, dec_rad, Dc, dT_ksz,
                                     row, bin_edges,
                                     vecdiff_ij_s, cij_s,
                                     dTw_s, w2_s, i_s, j_s,
                                     bin_number_s)

    vecdiff_ij_s = vecdiff_ij_s[:n_pairs]
    cij_s = cij_s[:n_pairs]
    dTw_s = dTw_s[:n_pairs]
    w2_s = w2_s[:n_pairs]
    i_s = i_s[:n_pairs]
    j_s = j_s[:n_pairs]
    bin_number_s = bin_number_s[:n_pairs]

    return vecdiff_ij_s, cij_s, dTw_s, w2_s, i_s, j_s, bin_number_s


def pairwise_given_rows_jitted(df, rows, bin_edges, MAX_NPAIRS=1000000):
    vecdiff_ij, cij, dTw, bin_number = [], [], [], []
    w2, i, j = [], [], []
    for row in rows:
        res = pairwise_one_row_and_save_all_jitted(df, row, bin_edges,
                                                   MAX_NPAIRS)
        vecdiff_ij.append(res[0])
        cij.append(res[1])
        dTw.append(res[2])
        w2.append(res[3])
        i.append(res[4])
        j.append(res[5])
        bin_number.append(res[6])
    vecdiff_ij = np.concatenate(vecdiff_ij)
    cij = np.concatenate(cij)
    dTw = np.concatenate(dTw)
    w2 = np.concatenate(w2)
    bin_number = np.concatenate(bin_number)
    i = np.concatenate(i)
    j = np.concatenate(j)

    data = {'i': i, 'j': j, 'cij': cij,
            'dTw': dTw, 'w2': w2, 'vecdiff': vecdiff_ij,
            'bin_number': bin_number}
    df = pd.DataFrame(data)
    return df


def compute_one_pairwise_chunk_saving_everything_to_lnx1032(df, N_groups,
        N_chunk,
        bin_edges, MAX_NPAIRS=1000000):  # noqa
    row_lists = np.array_split(np.arange(len(df)), N_groups)
    rows = row_lists[N_chunk]

    df_result = pairwise_given_rows_jitted(df, rows,
                                           bin_edges, MAX_NPAIRS)

    if not os.path.exists('/tmp/pag227'):
        os.mkdir('/tmp/pag227')
    fullpath = '/tmp/pag227/temp_data/'
    if not os.path.exists(fullpath):
        os.mkdir(fullpath)
    fname = os.path.join(fullpath, '%05i.hdf' % N_chunk)

    for bin_number in range(len(bin_edges)-1):
        df_towrite = df_result.query("bin_number == %i" % bin_number)
        df_towrite.to_hdf(fname, key='bin_number_%02i' % bin_number,
                          mode='a', format='table')

    os.system('ssh lnx1032-f1 mkdir /tmp/pag227/')
    os.system('ssh lnx1032-f1 mkdir /tmp/pag227/pairwise_data/')
    cmd = 'rsync -ravz %s lnx1032-f1:/tmp/pag227/pairwise_data' % fname  # noqa
    os.system(cmd)
    os.remove(fname)


#    #now transfer
#    client = paramiko.client.SSHClient()
#    client.load_system_host_keys()
#    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#    client.connect('lnx1032-f1')
#    sftp_client = client.open_sftp()
#  remote_file = sftp_client.open('/tmp/pag227/chunk_%05i.csv' % N_chunk, 'w')
#    df_result.to_hdf(remote_file, key='pairs', mode='w')
#    remote_file.close()
#    client.close()

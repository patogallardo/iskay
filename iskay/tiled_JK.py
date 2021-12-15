import numpy as np
import healpy as hp


def classify_grid(df, Nside):
    '''Receives a dataframe with ra, dec and bins it
    in a healpix grid.'''
    decs = df.dec.values
    ras = df.ra.values
    indx = hp.pixelfunc.ang2pix(Nside,
                                np.radians(-decs+90.),
                                np.radians(360.-ras))
    df1 = df.copy()
    df1['JK_index'] = indx
    return df1


def healpix_histogram_catalog(df1, Nside):
    '''Returns a vector in healpix format with
    per bin counts for the given catalog df.'''
    Npix = hp.nside2npix(Nside)
    df = df1.copy()
    m = np.zeros(Npix)
    df = classify_grid(df, Nside)
    cnts = df['JK_index'].value_counts()
    indices = cnts.index
    counts = cnts.values
    m[indices] = counts
    return m


def remove_edge_galaxies(df, tol_sigma, Nside):
    '''df: dataframe
    tol_sigma: tolerance in units of sigma, ex: 1.5
    Nside: Nside, power of 2
    '''
    df_labels = classify_grid(df, Nside=Nside)
    m = healpix_histogram_catalog(df_labels, Nside=Nside)
    sel = m != 0

    median = np.median(m[sel])
    std = np.sqrt(np.mean((m[sel] - median)**2))

    if median-std * tol_sigma < 1.0:  # too close to zero for a lower bound
        print("WARNING: the lower bound is too close to zero")
        print("Nside: %i" % Nside)

    sel_center = m > median - tol_sigma*std
    sel_edges = m < median - tol_sigma * std
    m_center = m.copy()
    m_center[sel_edges] = 0

    tokeep = np.isin(df_labels.JK_index.values, np.where(sel_center)[0])
    df_center = df_labels.loc[tokeep]

    return df_center


def how_many_tiles(df):
    return len(df.JK_index.unique())


def remove_tile(df, tile_indx):
    '''Removes the tiles with JK index given by tile_indx.
    tile_indx is the number of the ordered unique(JK_label).'''
    JK_indices = np.sort(df.JK_index.unique())
    assert tile_indx < len(JK_indices)
    df_to_return = df[df.JK_index != JK_indices[tile_indx]]
    return df_to_return


def getSide_given_iterations(df, Npix):
    '''For a given dataframe, gets the Npix value for a healpix grid
    that gives at least Nit tiles excluding edges.'''
    assert len(df) > Npix*3
    Nside = 2
    GO = True
    while GO:
        Nside = Nside * 2
        df_copy = df.copy()
        tiled = classify_grid(df_copy, Nside=Nside)
        tiled = remove_edge_galaxies(tiled, tol_sigma=1.5, Nside=Nside)
        Ntiles = how_many_tiles(tiled)
        GO = Ntiles < Npix
    if Ntiles > Npix * 1.5:
        Nside = Nside/2
    return Nside

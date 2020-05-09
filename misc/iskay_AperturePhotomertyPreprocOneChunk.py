'''aperture phtometry step for iskay.
Divide-and-conquer approach, meant to be run in a cluster.

Usage: first argument: chunkSize, chunkNumberToProcess.

Make sure that the environment variables OMP_NUM_THREADS and MKL_NUM_THREADS
are compatible with the number of cores you are requesting from the cluster.
Default for this code is to request 2 cores in
iskay_preprocessAperturePhotometry.py and have these environment variables
set to 2.

Written by: P. Gallardo.
'''
import numpy as np
from iskay import paramTools
from iskay import mapTools
from iskay import catalogTools
from iskay import submapTools
from iskay import cosmology
import os
import sys


def howManyEntries():
    '''Opens the catalog pointed to in params.ini and returns its length
    are there.'''
    params = paramTools.params('params.ini')
    df = catalogTools.cat(fname=params.CAT_FNAME).df
    return len(df)


def splitCatalog(fname, howManyGroups, chunk):
    '''Splits the dataframe in fname, into howManyGroups and returns
    the chunk number chunk.'''
    print "Extracting chunk: %i/%i" % (chunk, howManyGroups)
    df = catalogTools.cat(fname=fname).df
    chunk_to_return = np.array_split(df, howManyGroups)[chunk]
    print(chunk_to_return.head())
    print(chunk_to_return.tail())
    return chunk_to_return


def saveResult(thisChunk, T_disk, T_ring, div_disk, Dc, chunkNumber,
               masks=[None, None, None, None]):
    '''Adds two new columns to this chunk and uses pandas to save the
    resulting dataframe
    thisChunk: pandas dataframe with the original catalog without the ap
               photometry results
    T_dis, T_ring: vectors with the results from ap photo.
    chunkNumber: what chunk is this?
    masks: list that contains none if no info is to be included from mask
           otherwise it contains the numpy array with the masked disk
           photometry'''
    resultsFolder = 'ApPhotoResults'
    thisChunk['T_disk'] = T_disk
    thisChunk['T_ring'] = T_ring
    thisChunk['dT'] = T_disk - T_ring
    thisChunk['Dc'] = Dc
    thisChunk['div_disk'] = div_disk
    thisChunk['noise_muK'] = np.sqrt(1.0/div_disk)

    print "debug data in saveResult (before adding cols)"
    print(thisChunk.head())
    print(thisChunk.tail())

    for j in range(len(masks)):
        mask = masks[j]
        if mask is not None:
            thisChunk['masked_%i' % j] = mask
        else:
            thisChunk['masked_%i' % j] = np.nan
    print "debug data in saveResult (after adding cols)"
    print(thisChunk.head())
    print(thisChunk.tail())

    if not os.path.exists(resultsFolder):
        os.mkdir(resultsFolder)
    fname = os.path.join(resultsFolder,
                         'ApPhotoCat_%03i.csv' % chunkNumber)
    print(fname)
    thisChunk.to_csv(fname)


def extractApPhotometry(params, thisChunk, mapOrDiv='map'):
    '''Wrapper for getApPhotometryForCatalogPositions.
    Receives fname for map, opens it and runs
    getApPhotometryForCatalogPositions'''
    mapfname = {'map': params.FITS_FNAME,
                'div': params.DIVMAP_FNAME,
                'mask1': params.MASKMAP_FNAME1,
                'mask2': params.MASKMAP_FNAME2,
                'mask3': params.MASKMAP_FNAME3,
                'mask4': params.MASKMAP_FNAME4}[mapOrDiv]
    if 'mask' in mapOrDiv:
        mapOrDiv = 'mask'  # all params for masks are the same
    repixelize = {'map': params.REPIXELIZE,
                  'div': False,
                  'mask': False}[mapOrDiv]
    reproject = {'map': params.REPIXELIZE,
                 'div': False,
                 'mask': False}[mapOrDiv]
    photoDiskR = {'map': params.PHOTODISKR,
                  'div': params.PHOTORINGR,
                  'mask': params.PHOTORINGR*1.4}[mapOrDiv]
    photoRingR = {'map': params.PHOTORINGR,
                  'div': params.PHOTORINGR*1.4,
                  'mask': params.PHOTORINGR*1.4*1.4}[mapOrDiv]

    theMap = mapTools.openMap_remote(fname=mapfname)
    ras_deg, decs_deg = thisChunk['ra'].values, thisChunk['dec'].values
    T_disk, T_ring = submapTools.getApPhotometryForCatalogPositions(theMap,
                      ras_deg, decs_deg,  # noqa
                      photoDiskR, photoRingR,
                      repixelize=repixelize,
                      reprojection=reproject,
                      silent=False)
    return T_disk, T_ring


catalogLength = howManyEntries()
chunkSize = int(sys.argv[1])
chunkNumber = int(sys.argv[2])  # which group to process?
howManyGroups = np.ceil(float(catalogLength)/float(chunkSize))
assert chunkNumber < howManyGroups  # starts at 0 and ends
# in howManyGroups - 1

params = paramTools.params('params.ini')
thisChunk = splitCatalog(fname=params.CAT_FNAME,
                         howManyGroups=howManyGroups,
                         chunk=chunkNumber)

T_disk, T_ring = extractApPhotometry(params, thisChunk, mapOrDiv='map')
print "Done extracting the map, loading the divmap now..."
div_disk, div_ring = extractApPhotometry(params, thisChunk, mapOrDiv='div')


mask_disk1 = None
if '.fits' in params.MASKMAP_FNAME1:
    print('Loading mask1 now')
    mask_disk1, mask_ring1 = extractApPhotometry(params,
                                                 thisChunk, mapOrDiv='mask1')
mask_disk2 = None
if '.fits' in params.MASKMAP_FNAME2:
    print('Loading mask2 now')
    mask_disk2, mask_ring2 = extractApPhotometry(params,
                                                 thisChunk, mapOrDiv='mask2')
mask_disk3 = None
if '.fits' in params.MASKMAP_FNAME3:
    print('Loading mask3 now')
    mask_disk3, mask_ring3 = extractApPhotometry(params,
                                                 thisChunk, mapOrDiv='mask3')
mask_disk4 = None
if '.fits' in params.MASKMAP_FNAME4:
    print('Loading mask4 now')
    mask_disk4, mask_ring4 = extractApPhotometry(params,
                                                 thisChunk, mapOrDiv='mask4')
masks = [mask_disk1, mask_disk2, mask_disk3, mask_disk4]

Dc = cosmology.Dc(thisChunk.z.values)
saveResult(thisChunk, T_disk, T_ring, div_disk, Dc, chunkNumber,
           masks=masks)

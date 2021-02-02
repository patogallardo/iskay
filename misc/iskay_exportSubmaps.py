'''
Usage:
    exportSubmaps dsetName howMany
    dsetName: can be 'submaps' or 'divmap'
    howMany: can be nothing in which case whole catalog is processed. If
             a number is given it is the length of the catalog to process.
             This is useful for debugging.
'''
from iskay import paramTools
from iskay import mapTools
from iskay import catalogTools
from iskay import submapTools
import numpy as np
from pixell import enmap
from pixell import reproject
import h5py
import os
import sys
import progressbar

assert len(sys.argv) >= 2  # must at least say dsetname
dsetName = sys.argv[1]
assert dsetName in ['submaps', 'divmaps']
howMany = None  # how many objects to process, Use None to process entire cat.
if len(sys.argv) == 3:
    howMany = int(sys.argv[2])


def extractStamp(submap, ra_deg, dec_deg, r_ring_arcmin, repixelize=True,
                 reprojection=True):
    extractedSubmap = submap.copy()
    if repixelize:
        extractedSubmap = enmap.resample(extractedSubmap,
                                         10 * np.array(submap.shape))
    if reprojection:
        extractedSubmap = reproject.postage_stamp(extractedSubmap,
                                                  ra_deg, dec_deg,
                                                  2*r_ring_arcmin,
                                                  0.5/5.)[0, :, :]
    else:  # this was added for backwards compatibility
        assert len(extractedSubmap.shape) == 2
        extractedSubmap = submapTools.getSubmap_originalPixelization(
                                theMap=extractedSubmap, # noqa
                                ra_deg=ra_deg, dec_deg=dec_deg, # noqa
                                semiWidth_deg=r_ring_arcmin/60.) # noqa
    return extractedSubmap


def writeSubapsToFile(theMap, df,
                      photoringR_arcmin,
                      params=None,
                      dsetName=None,
                      verbose=False):
    submapSemiWidthR_arcmin = photoringR_arcmin * 3

    howMany = len(df)
    ras_deg, decs_deg = df['ra'].values, df['dec'].values
    assert params is not None
    assert dsetName is not None
#do it once to check array dimensions
    submap = submapTools.getSubmap_originalPixelization(theMap,
                                                    ras_deg[0], decs_deg[0],
                                         4.0*photoringR_arcmin/60.)  # noqa
    sampleStamp = extractStamp(submap, ras_deg[0], decs_deg[0],
                               submapSemiWidthR_arcmin,
                               repixelize=params.REPIXELIZE,
                               reprojection=params.REPROJECT)
    shape = sampleStamp.shape

    outputDir = '/tmp/pag227/ApPhotoResults'
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)
    f = h5py.File('%s/%s_%s.h5' % (outputDir,
                                   params.FITS_FNAME.split('/')[-1],
                                   dsetName), 'w')
    dset = f.create_dataset('%s' % dsetName, (howMany, shape[0], shape[1]),
                            dtype='float16')
    modrmap = f.create_dataset('modr', (shape[0], shape[1]))
    modrmap[:, :] = np.array(sampleStamp.modrmap())

    widgets = [progressbar.Percentage(),
               progressbar.Bar(),
               progressbar.ETA()]
    if verbose:
        bar = progressbar.ProgressBar(widgets=widgets,  # show bar?
                                      max_value=howMany)
        bar.start()
    for j in xrange(howMany):
        print j
        ra, dec = ras_deg[j], decs_deg[j]
        submap = submapTools.getSubmap_originalPixelization(theMap,
                                                        ra, dec,
                                              4*photoringR_arcmin/60.)  # noqa
        stamp = extractStamp(submap, ra, dec, submapSemiWidthR_arcmin,
                             repixelize=params.REPIXELIZE,
                             reprojection=params.REPROJECT)
        dset[j, :, :] = np.array(stamp)
        if verbose:
            bar.update(j+1)
    if verbose:
        bar.finish()
    f.close()


if howMany is not None:
    print "Processing only %i objects, use howMany=None full cat" % howMany
params = paramTools.params('params.ini')
mapfnames = {'submaps': params.FITS_FNAME, 'divmaps': params.DIVMAP_FNAME}

theMap = mapTools.openMap_remote(fname=mapfnames[dsetName])
df = catalogTools.cat(fname=params.CAT_FNAME,
                      howMany=howMany).df

writeSubapsToFile(theMap, df, params.PHOTORINGR, params,
                  dsetName=dsetName, verbose=True)

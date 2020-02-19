'''Tools to deal with submaps from an enmap.
'''
import numpy as np
from pixell import enmap
from pixell import reproject
import progressbar


#make a test for this
def getApPhotometryForCatalogPositions(theMap, ras_deg, decs_deg,
                                       r_disk_arcmin, r_ring_arcmin,
                                       repixelize=False, reprojection=False,
                                       submapsFilename=None, indices=None,
                                       silent=True):
    '''theMap: enmap object with the map
        ras_deg, decs_deg: arrays of ra/dec positions in degrees
        r_disk_arcmin, r_ring_arcmin: r_disk and r_ring in arcmins
        repixelize: True for resampling to higher res before postage stamp
        reprojeciton: True for using postage stamp to take care of projection
        submapsFilename: None if you dont want to save submaps. In other
                         case, it is the filename of the file that contains
                         the submaps.
        indices: index from the original catalog, to identify each galaxy,
                 in the future this can be the object ID.
    '''
    assert len(ras_deg) == len(decs_deg)
    howMany = len(ras_deg)
    T_disks, T_rings = np.empty(ras_deg.shape), np.empty(decs_deg.shape)
    semiWidth_deg = 4.0 * r_ring_arcmin/60.

    if not silent:
        widgets = [progressbar.Percentage(), progressbar.Bar(),
                   progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets,
                                      max_value=howMany).start()

    for j in xrange(howMany):
        ra_deg, dec_deg = ras_deg[j], decs_deg[j]
        submap = getSubmap_originalPixelization(theMap,
                                                ra_deg, dec_deg,
                                                semiWidth_deg)

        T_disks[j], T_rings[j] = get_aperturePhotometry(submap,
                                                        ra_deg, dec_deg,
                                                        r_ring_arcmin,
                                                        r_disk_arcmin,
                                                        repixelize=repixelize,
                                    reprojection=reprojection) # noqa
        if not silent:
            bar.update(j+1)
    if not silent:
        bar.finish()
    return T_disks, T_rings


#TODO: write a test for this
def get_aperturePhotometry(submap, ra_deg, dec_deg,
                           r_ring_arcmin, r_disk_arcmin,
                           repixelize=True, reprojection=True):
    '''Gets T_disk and T_ring for a given submap. Needs ra and dec of
    the center of the submap and the radius of the ring and disk for the
    aperture.
    Can repixelize by a factor of 10 finer resolution and reproject.
    submap: submap in the original pixelization
    ra, dec_deg: ra and dec in degrees of the center of the aperture
    r_ring/disk_arcmin: radius for the aperture photometry
    repixelize: True if you want to do it as enmap.resample by a factor of 10
    reprojection: True if you want to extract the postage stamp by
    reprojection at the equator.
    submapsFilename: None if you dont want to save submaps. String with a
    hdf5 filename if you want to save the submaps.'''
    submapForPhotometry = submap.copy()
    if repixelize:
        submapForPhotometry = enmap.resample(submapForPhotometry,
                                             10*np.array(submap.shape))
    if reprojection:
        submapForPhotometry = reproject.postage_stamp(submapForPhotometry,
                                                      ra_deg, dec_deg,
                                                      3.0*r_ring_arcmin,
                                                      0.5/10.)[0, :, :]
    r_arcmin = 60. * np.rad2deg(submapForPhotometry.modrmap())
    sel_disk = r_arcmin < r_disk_arcmin
    sel_ring = np.logical_and(r_arcmin > r_disk_arcmin,
                              r_arcmin < r_ring_arcmin)

    T_disk = submapForPhotometry[sel_disk].mean()
    T_ring = submapForPhotometry[sel_ring].mean()
    return T_disk, T_ring


def gen_boxes(ra_deg, dec_deg, semiWidth_deg):
    '''
    ra_deg: ra center in degrees
    dec_deg: dec center in degrees
    semiWidth_deg: semiwidth in degrees.
    Uses enmap.submap to get a submap without changing the pixelization.
    returns a box in radians with [[decmin, ramin],[decmax, ramax]]
    that can be used by enmap.
    '''
    ra_deg = np.array(ra_deg)
    dec_deg = np.array(dec_deg)

    howMany = ra_deg.shape[0]
    boxes = np.empty([howMany, 2, 2])

    boxes[:, 0, 0] = dec_deg - semiWidth_deg
    boxes[:, 0, 1] = ra_deg - semiWidth_deg
    boxes[:, 1, 0] = dec_deg + semiWidth_deg
    boxes[:, 1, 1] = ra_deg + semiWidth_deg

    return np.deg2rad(boxes)


def gen_box(ra_deg, dec_deg, semiWidth_deg):
    '''Gets ra, dec for one source and returns the box for it.
    ra_deg, dec_deg: number in degrees.
    semiWidth_deg: number in degrees.'''
    box = np.empty([2, 2])

    box[0, 0] = dec_deg - semiWidth_deg
    box[0, 1] = ra_deg - semiWidth_deg
    box[1, 0] = dec_deg + semiWidth_deg
    box[1, 1] = ra_deg + semiWidth_deg

    return np.deg2rad(box)


#changing behaviour to accept a filename and do its thing...
#remember to review tests
def getSubmaps_originalPixelization(theMap, ras_deg, decs_deg,
                                    semiWidth_deg):
    '''From the original pixelization of the map, extract a list of submaps.
    theMap_fname: map to extract submaps from
    ra_deg, dec_deg: arrays of ra and decs in degrees
    semiWidth_deg: half of the width of the submap to be extracted in degrees.
    '''
    boxes = gen_boxes(ras_deg, decs_deg, semiWidth_deg)
    submaps = []
    for box in boxes:
        submaps.append(enmap.submap(theMap, box))
    return submaps


#test this
def getSubmap_originalPixelization(theMap, ra_deg, dec_deg, semiWidth_deg):
    '''Receives theMap and gets a submap centered in ra,dec with width
    semiWidth_deg'''
    box = gen_box(ra_deg, dec_deg, semiWidth_deg)
    submap = enmap.submap(theMap, box)
    return submap


def resampleSubmaps(submapList, resampleFactor=10):
    '''Gets a list of enmap submaps and resamples them by a constant factor
    Uses defaults in enmap.resample, fft, and order=3
    '''
    resampledSubmapList = []
    for submap in submapList:
        newShape = resampleFactor * np.array(submap.shape, dtype=int)
        resampledSubmapList.append(enmap.resample(submap, newShape))
    return resampledSubmapList

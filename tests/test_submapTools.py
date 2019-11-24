from iskay import submapTools
from iskay import mapTools
import numpy as np
from pixell import enmap

map_fname = 'f150_daynight_all_map_mono.fits'
theMap = mapTools.openMap_remote(map_fname)
# we will use this in a lot of tests


def test_getSubmap_originalPixelization():
    ra_deg = np.random.uniform(low=0, high=360)
    dec_deg = np.random.uniform(low=-40, high=0)
    semiWidth_deg = 0.2
    box = submapTools.gen_box(ra_deg, dec_deg, semiWidth_deg)
    submap = submapTools.getSubmap_originalPixelization(theMap,
                                                        ra_deg,
                                                        dec_deg,
                                                        semiWidth_deg)
    submap_noReproject = enmap.submap(theMap, box)
    diff_sq = (submap - submap_noReproject)**2
    assert diff_sq.sum() == 0


def test_gen_box():
    ra_deg = np.random.uniform(low=0, high=360)
    dec_deg = np.random.uniform(low=-30, high=0)
    semiWidth_deg = 0.2
    res_genbox = submapTools.gen_box(ra_deg, dec_deg, semiWidth_deg)

    ras_deg = np.array([ra_deg])
    decs_deg = np.array([dec_deg])
    res_genboxes = submapTools.gen_boxes(ras_deg, decs_deg, semiWidth_deg)

    diff_sq = (res_genbox - res_genboxes[0])**2
    assert diff_sq.sum() < 1e-6


def test_gen_boxes():
    ra_deg = np.array([0, 1])
    dec_deg = np.array([2, 3])
    semiwidth_deg = 10

    fiducial = [[
                [dec_deg[0]-semiwidth_deg, ra_deg[0]-semiwidth_deg],
                [dec_deg[0]+semiwidth_deg, ra_deg[0]+semiwidth_deg]],
                [[dec_deg[1]-semiwidth_deg, ra_deg[1]-semiwidth_deg],
                [dec_deg[1]+semiwidth_deg, ra_deg[1]+semiwidth_deg]]
                ]
    fiducial = np.deg2rad(fiducial)
    res = submapTools.gen_boxes(ra_deg, dec_deg, semiwidth_deg)
    difsq = np.sum((fiducial-res)**2)
    assert difsq < 1e-15


def test_getSubmaps_originalPixelization():
    ra_deg = np.array([0, 1])
    dec_deg = np.array([2, 3])
    semiWidth_deg = 0.5
    submaps = submapTools.getSubmaps_originalPixelization(theMap,
                                                          ra_deg,
                                                          dec_deg,
                                                          semiWidth_deg)
    assert len(ra_deg) == len(submaps)

    for j, submap in enumerate(submaps):
        center = np.rad2deg(submap.center())
        diff_center_sq = (center - np.array([dec_deg[j], ra_deg[j]]))**2
        assert diff_center_sq.sum() < 1e-3  # check centers
        extent = np.rad2deg(submap.extent())
        diff_extent_sq = (extent-2*semiWidth_deg)**2
        assert diff_extent_sq.sum() < 1e-3  # check extent


def test_resampleSubmaps():
    size = 300
    resampleFactor = 10
    ra_deg = np.random.uniform(low=0, high=300, size=size)
    dec_deg = np.random.uniform(low=-30, high=0, size=size)

    submaps = submapTools.getSubmaps_originalPixelization(theMap,
                                                          ra_deg,
                                                          dec_deg,
                                                          0.25)
    resampled = submapTools.resampleSubmaps(submaps,
                                            resampleFactor=resampleFactor)
    assert len(resampled) == len(ra_deg)
    for j in xrange(len(submaps)):
        shape_resampled = np.array(resampled[j].shape)
        shape_submap = np.array(submaps[j].shape)
        assert np.all(shape_resampled - resampleFactor * shape_submap == 0)

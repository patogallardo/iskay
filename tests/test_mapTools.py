from iskay import mapTools

map_fname = 'f150_daynight_all_map_mono.fits'


def test_openMap_local():
    theMap = mapTools.openMap_local(map_fname)
    assert len(theMap.shape) == 2


def test_openMap_remote():
    theMap1 = mapTools.openMap_local(map_fname)
    theMap2 = mapTools.openMap_remote(map_fname)

    diff_sq = (theMap1 - theMap2)**2
    assert diff_sq.sum() < 1e-9

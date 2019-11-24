import os
from iskay import envVars
from iskay import transferDataTools
from pixell import enmap


#todo write a test.
def openMap_remote(fname):
    '''Opens remotely the map.
       Returns the enmap object.
    '''
#    transferDataTools.checkIfExistsAndCopyIfNot(fname)
    fitsFullPath = os.path.join(envVars.remotePath1, fname)
    print("Loading map remotely")
    print "Loading fits file: %s..." % fitsFullPath
    fitsMap = enmap.read_fits(fitsFullPath)
    if len(fitsMap.shape) == 3:
        fitsMap = fitsMap[0, :, :]  # temp map
    return fitsMap


def openMap_local(fname):
    '''Checks if map exists and dowloads it.
    Returns the enmap object.
    '''
    transferDataTools.checkIfExistsAndCopyIfNot(fname)
    fitsFullPath = os.path.join(envVars.localDataPath, fname)
    print("Loading map locally...")
    print "Loading fits file: %s..." % fitsFullPath
    fitsMap = enmap.read_fits(fitsFullPath)  # temp map
    if len(fitsMap.shape) == 3:
        fitsMap = fitsMap[0, :, :]
    return fitsMap

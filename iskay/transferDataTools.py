'''Tools to make the transfer of data easy.'''
import os
import envVars
import shutil


def checkIfExistsAndCopyIfNot(fname, debug=True):
    '''Looks into localDataPath and checks if data exists.
    If it doesn't it will try to fetch data from remotePath1 and
    remotePath2
    fname: file name to transfer. note this is not a full path, just
    the filename'''
    localFullPath = os.path.join(envVars.localDataPath, fname)
    remote1FullPath = os.path.join(envVars.remotePath1, fname)
    remote2FullPath = os.path.join(envVars.remotePath2, fname)

    existsLocally = os.path.exists(localFullPath)
    if existsLocally:
        print "file exists locally but will be overwritten"
        existsLocally = False  # always rewrite to make sure we are in sync
    existsInRemotePath1 = os.path.exists(remote1FullPath)
    existsInRemotePath2 = os.path.exists(remote2FullPath)
    if existsLocally:
        if debug:
            print("file %s exists locally" % fname)
        return True
    elif existsInRemotePath1:
        print("Found file %s in remotePath1, transfering..." % fname)
        shutil.copy(remote1FullPath, envVars.localDataPath)
        print('Done Transfering File: %s' % fname)
        return True
    elif existsInRemotePath2:
        print("Found file %s in remotePath2, transfering..." % fname)
        shutil.copy(remote2FullPath, envVars.localDataPath)
        print('Done Transfering File: %s' % fname)
        return True
    print('Cound not find file %s' % fname)

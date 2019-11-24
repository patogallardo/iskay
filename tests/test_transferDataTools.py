import os
import iskay.envVars as envVars
import iskay.transferDataTools


def touchFile(fullpath):
    assert fullpath.find('.') != -1
    open(fullpath, 'a').close()


def deleteFile(fullpath):
    assert fullpath.find('.') != -1
    os.unlink(fullpath)


def test_DoNothingIfDataExists():
    '''Writes a test file in the local folder and checks if
    transferDataTools can see it'''
    fullpath = os.path.join(envVars.localDataPath, 'test.txt')
    touchFile(fullpath)
    iskay.transferDataTools.checkIfExistsAndCopyIfNot('test.txt')


def test_copyIfDataInLocalDataPath2():
    '''Remove local copy of test.txt and see if we can trasnfer it
    from a remotePath.'''
    deleteFile(os.path.join(envVars.localDataPath, 'test.txt'))
    fullpath = os.path.join(envVars.remotePath2, 'test.txt')
    touchFile(fullpath)
    iskay.transferDataTools.checkIfExistsAndCopyIfNot('test.txt')
    deleteFile(os.path.join(envVars.localDataPath, 'test.txt'))
    deleteFile(fullpath)

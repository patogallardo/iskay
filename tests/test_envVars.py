import iskay.envVars as envVars
import os


def test_localDataPath():
    assert os.path.exists(envVars.localDataPath)


def test_remoteDataPaths():
    firstDirExists = os.path.exists(envVars.remotePath1)
    secondDirExists = os.path.exists(envVars.remotePath2)
    assert firstDirExists and secondDirExists

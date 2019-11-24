from iskay import catalogTools
import os
import numpy as np
import shutil
from iskay import envVars

testPath = '/'.join((catalogTools.__file__).split('/')[:-2])+'/tests/'
testCatalogFullPath = os.path.join(testPath, 'data_toTestAPI/testCat.csv')

shutil.copy(testCatalogFullPath, envVars.remotePath2)


def testOpenCat_NoArguments():
    fname = 'testCat.csv'
    catalog = catalogTools.cat(fname)
    assert len(catalog.df) > 10


def testOpenCat_query():
    fname = 'testCat.csv'
    query = 'ra>30 and dec>0 and z>0.1'
    catalog = catalogTools.cat(fname, query=query)
    ra_query_ok = catalog.df.ra.min() > 30
    dec_query_ok = catalog.df.dec.min() > 0
    z_query_ok = catalog.df.z.min() > 0.1
    assert ra_query_ok and dec_query_ok and z_query_ok


def testOpenCat_sortBy():
    fname = 'testCat.csv'
    sortBy = 'z'
    catalog = catalogTools.cat(fname, sortBy=sortBy)
    #check if largest number is the first
    max_ok = catalog.df.z.iloc[0] == catalog.df.z.values.max()
    min_ok = catalog.df.z.iloc[-1] == catalog.df.z.values.min()
    assert max_ok and min_ok


def testOpenCat_howMany():
    fname = 'testCat.csv'
    howMany = 30
    catalog = catalogTools.cat(fname, howMany=howMany)
    assert len(catalog.df) == howMany


def test_preProcessedCat():
    directory = os.path.join(testPath, 'data_toTestAPI', 'ApPhotoResults')
    pattern = 'testPreprocessedCat_*.csv'
    cat = catalogTools.preProcessedCat(pattern, directory)
    assert len(cat.df) > 1


def test_preProcessedCat_query():
    query = 'ra>20.0'
    directory = os.path.join(testPath, 'data_toTestAPI', 'ApPhotoResults')
    pattern = 'testPreprocessedCat_*.csv'
    df = catalogTools.preProcessedCat(pattern, directory, query=query).df
    assert np.all(df.ra.values > 20)


def test_preProcessedCat_sortBy():
    sortby = 'ra'
    directory = os.path.join(testPath, 'data_toTestAPI', 'ApPhotoResults')
    pattern = 'testPreprocessedCat_*.csv'
    df = catalogTools.preProcessedCat(pattern, directory, sortBy=sortby).df
    assert (df.ra.values.max() == df.ra.values[0])


def test_preProcessedCat_howMany():
    howMany = 14
    directory = os.path.join(testPath, 'data_toTestAPI', 'ApPhotoResults')
    pattern = 'testPreprocessedCat_*.csv'
    df = catalogTools.preProcessedCat(pattern, directory, howMany=howMany).df
    assert howMany == len(df.ra.values)

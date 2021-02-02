'''Tools for loading a catalog from csv file.

At opening, the class cat has amenities to retrieve and open file.
It also can sort, select the first N elements and apply an arbitrary
query in pandas style.'''
from iskay import transferDataTools
import glob
import os
import pandas as pd
import numpy as np


class cat(object):
    def __init__(self, fname,
                 sortBy=None, query=None, howMany=None):
        '''
        fname: fname of csv file.
        sortBy: None: do nothing. string: name of the column to sort by.
                Sorting will be done in descending order. Largest first
                (zero index will have the largest values)
        query: pandas query to apply to the dataframe once open
        howMany: how many columns to take from the data (if you want
                a smaller sample. Use this in conjunction to sortBy.
        '''
        self.fname = fname

        fname_fullpath = transferDataTools.searchCatalog(fname)

        self.fname_fullpath = fname_fullpath
        df = pd.read_csv(fname_fullpath, comment='#')

        # patch added on 20200602 to conform to rad, decd, zd, lumd naming
        # of columns
        if 'rad' in df.columns:
            df.rename(columns={'rad': 'ra',
                               'decd': 'dec',
                               'lumd': 'lum',
                               'zd': 'z',
                               'idd': 'id'}, inplace=True)
            print("renamed cols")

        #apply query
        if query is not None:
            df.query(query, inplace=True)
        #sort by a column
        if sortBy is not None:  #largest values first
            df.sort_values(sortBy, inplace=True, ascending=False)
        #how many rows to open
        if howMany is not None:
            df = df.head(howMany)  #first 'howMany' samples
        self.df = df


def openPreprocessedDataFollowingPattern(pattern, directory='ApPhotoResults'):
    '''Opens a lot of csv files and appends them returning a df.
    See class postProcCat for details
    directory sets where the data lives.'''
    fnames = glob.glob(os.path.join(directory, pattern))
    fnames.sort()
    df = pd.concat([pd.read_csv(f, index_col=0) for f in fnames])
    assert np.sum(np.diff(df.index.values)-1) == 0  # check csv continuity
    return df


class preProcessedCat(object):
    def __init__(self, pattern='ApPhotoCat_*.csv', directory='ApPhotoResults',
                 sortBy=None, howMany=None, query=None):
        '''Opens directory ApPhotoResults and looks for the pattern
        ApPhotoCat_*.csv, pattern can be changed via variable pattern'''
        assert os.path.exists(directory)  # directory must exist
        df = openPreprocessedDataFollowingPattern(pattern, directory)
        #apply query
        if query is not None:
            df.query(query, inplace=True)
        #sort by a column
        if sortBy is not None:  #largest values first
            df.sort_values(sortBy, inplace=True, ascending=False)
        #how many rows to open
        if howMany is not None:
            df = df.head(howMany)  #first 'howMany' samples
        self.df = df

'''Tools for reading parameters from .ini file.

All the action happens in the params object, which receives a filename for
the .ini file. '''

import ConfigParser
import numpy as np


class params:
    def __init__(self, fname='params.ini'):
        '''Receives a fname for a parameter file.
        Stores the variables contained in the file in this object.'''
        c = ConfigParser.ConfigParser()
        c.read(fname)
        self.config = c
        checkConfigParserFile(c)  # check sections
        self.readParams(c)

    def readParams(self, c):
        '''Gets and loads to self the parameters we want to store from
        the params file, every parameter must have a variable created.'''
        self.NAME = c.get('Name', 'analysisname')
        self.FITS_FNAME = c.get('Map', 'fits_fname')
        self.DIVMAP_FNAME = c.get('Map', 'divmap_fname')
        self.MASKMAP_FNAME1 = c.get('Map', 'maskmap_fname1')
        self.MASKMAP_FNAME2 = c.get('Map', 'maskmap_fname2')
        self.MASKMAP_FNAME3 = c.get('Map', 'maskmap_fname3')
        self.MASKMAP_FNAME4 = c.get('Map', 'maskmap_fname4')

        self.CAT_FNAME = c.get('Catalog', 'catalog_fname')

        query = c.get('Catalog', 'query')
        self.CAT_QUERY = query
        if query == '' or query == ' ' or query == 'None':
            self.CAT_QUERY = None
        self.N_OBJ = c.getint('AnalysisParams', 'n_obj')
        self.PHOTODISKR = c.getfloat('AnalysisParams', 'photodiskr')
        self.PHOTORINGR = c.getfloat('AnalysisParams', 'photoringr')
        self.SIGMA_Z = c.getfloat('AnalysisParams', 'sigma_z')

        #following lines check if binning is constant size or adaptive
        if c.get('AnalysisParams', 'bin_size_mpc') == 'None':
            self.BIN_SIZE_MPC = None
        else:
            self.BIN_SIZE_MPC = c.getfloat('AnalysisParams', 'bin_size_mpc')

        if c.get('AnalysisParams', 'n_bins') == 'None':
            self.N_BINS = None
        else:
            self.N_BINS = c.getint('AnalysisParams', 'n_bins')

        #adaptive bins
        if c.get('AnalysisParams', 'bin_edges_mpc') == 'None':
            self.BIN_EDGES = None
            self.UNEVEN_BINS = False
        else:  # separate by comma space: 1, 2, 3
            bin_edges_str = c.get('AnalysisParams', 'bin_edges_mpc').split(',') # noqa
            bin_edges = np.array([float(element) for element in bin_edges_str])
            self.BIN_EDGES = np.sort(bin_edges)
            self.UNEVEN_BINS = True
        #check that bins are defined uniformly or unevenly spaced
        assert (((self.BIN_SIZE_MPC is None) and (self.N_BINS is None))
                ^ (self.BIN_EDGES is None)) # noqa exclusively one

        self.GET_TZAV_FAST = c.getboolean('AnalysisParams', 'get_tzav_fast')
        self.DO_VARIANCE_WEIGHTED = c.getboolean('AnalysisParams',
                                                 'do_variance_weighted')
        self.JK_NGROUPS = c.getint('JK', 'n_groups')
        self.JK_RESAMPLING_METHOD = c.get('JK', 'resampling_method')
        assert self.JK_RESAMPLING_METHOD in ['JK', 'BOOTSTRAP',
                                             'jk', 'bootstrap',
                                             'bootstrap_pairwise',
                                             'bs_dt']

        self.REPIXELIZE = c.getboolean('submap', 'repixelize')
        self.REPROJECT = c.getboolean('submap', 'reproject')


def checkConfigParserFile(c):
    '''Makes some basic tests to see if the file has the minimum.'''
    mustHaves = ['Name', 'Map', 'Catalog', 'AnalysisParams', 'JK']
    for mustHave in mustHaves:
        assert mustHave in c.sections()


def generateDefaultParams():
    '''Creates a params object with the default configuration.
    This is used to generate an example params.ini file.
    '''
    c = ConfigParser.ConfigParser()
    c.add_section('Name')
    c.set('Name', 'AnalysisName', 'name')
    c.add_section('Map')
    c.set('Map', 'FITS_FNAME', 'act_planck_f150_map_mono.fits')
    c.set('Map', 'DIVMAP_FNAME', 'act_planck_f150_div_mono.fits')
    c.set('Map', 'MASKMAP_FNAME1', 'None')
    c.set('Map', 'MASKMAP_FNAME2', 'None')
    c.set('Map', 'MASKMAP_FNAME3', 'None')
    c.set('Map', 'MASKMAP_FNAME4', 'None')

    c.add_section('Catalog')
    c.set('Catalog', 'CATALOG_FNAME',
          'DR15_actplanck_catalog_wbestObjID_20190501_EMV_evavagiakis_kcorrected_ra_dec_z_lum_id_CUTS_20190617.csv')  # noqa
    c.set('Catalog', 'QUERY', '')

    c.add_section('AnalysisParams')
    c.set('AnalysisParams', 'N_OBJ', '1000000')
    c.set('AnalysisParams', 'PhotoDiskR', '2.1')
    c.set('AnalysisParams', 'PhotoRingR', '%s' % (2.1 * 1.4))
    c.set('AnalysisParams', 'Sigma_z', '0.01')
    c.set('AnalysisParams', 'bin_size_mpc', 'None')
    c.set('AnalysisParams', 'n_bins', 'None')
    c.set('AnalysisParams', 'bin_edges_mpc', '0.0, 10.0, 20.0, 30.0, 40.0, '
          '50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, '
          '150.0, 200.0, 250.0, 315.0, 395.0')
    c.set('AnalysisParams', 'get_tzav_fast', 'True')
    c.set('AnalysisParams', 'do_variance_weighted', 'False')

    c.add_section('JK')
    c.set('JK', 'n_groups', '500')
    c.set('JK', 'resampling_method', 'JK')

    c.add_section('submap')
    c.set('submap', 'repixelize', 'True')
    c.set('submap', 'reproject', 'True')

    return c

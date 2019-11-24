import iskay.paramTools as paramTools
import os

testPath = '/'.join((paramTools.__file__).split('/')[:-2])+'/tests/'
testConfigFileFullPath = os.path.join(testPath, 'data_toTestAPI/params.ini')


def test_openConfigFile():
    pars = paramTools.params(testConfigFileFullPath)
    assert type(pars.CAT_FNAME) is str
    assert type(pars.CAT_QUERY) is str
    assert type(pars.FITS_FNAME) is str
    assert type(pars.N_OBJ) is int
    assert type(pars.NAME) is str
    assert type(pars.PHOTODISKR) is float
    assert type(pars.PHOTORINGR) is float
    assert type(pars.REPROJECT) is bool
    assert type(pars.REPIXELIZE) is bool
    assert type(pars.JK_NGROUPS) is int
    assert type(pars.N_BINS) is int
    assert type(pars.BIN_SIZE_MPC) is float
    assert type(pars.SIGMA_Z) is float


def test_generateDefaultParams():
    c = paramTools.generateDefaultParams()
    sections = c.sections()
    assert 'Map' in sections
    assert 'Name' in sections and 'Catalog' in sections
    assert 'AnalysisParams' in sections and 'JK' in sections

    assert c.get('Name', 'AnalysisName').find('') > -1

    assert c.get('Map', 'fits_fname').find('.fits') > -1
    assert c.get('Map', 'divmap_fname').find('.fits') > -1

    assert c.get('Catalog', 'catalog_fname').find('.csv') > -1
    assert c.get('Catalog', 'query') == ''

    assert type(c.getint('AnalysisParams', 'n_obj')) is int
    assert c.getfloat('AnalysisParams', 'photodiskr') == 2.1
    assert c.getfloat('AnalysisParams', 'photoringr') == 2.94

    assert c.getint('JK', 'n_groups') == 50

    assert c.getboolean('submap', 'repixelize') is True
    assert c.getboolean('submap', 'reproject') is True

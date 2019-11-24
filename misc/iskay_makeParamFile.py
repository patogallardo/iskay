#!/usr/bin/env python2

'''Generates an example param file that is functional and can
be edited to point to the actual analysis.'''

import iskay.paramTools as paramTools
import iskay
import os
import shutil

miscFolderPath = '/'.join(iskay.__file__.split('/')[:-2]) + '/misc/'


def printWelcomeMsg():
    '''This does nothing just show what we are doing'''
    print('Generating a parameter file for iskay.')


def writeParamsFile(fname, config):
    '''Writes a configuration object config into filename fname.
    fname: string with the filename to write.
    config: ConfigParser object to write.
    '''
    with open(fname, 'w') as f:
        config.write(f)


def copyScripts():
    '''Copies example scripts from library to current directory. '''
    origin = os.path.join(miscFolderPath, 'iskay_analysis.py')
    destination = './'
    shutil.copy(origin, destination)
    origin = os.path.join(miscFolderPath,
                          'iskay_preprocessAperturePhotometry.py')
    destination = './'
    shutil.copy(origin, destination)
    origin = os.path.join(miscFolderPath,
                          'qsub_exportSubmaps.sh')
    destination = './'
    shutil.copy(origin, destination)


fname = 'params.ini'


c = paramTools.generateDefaultParams()
writeParamsFile(fname, c)
copyScripts()

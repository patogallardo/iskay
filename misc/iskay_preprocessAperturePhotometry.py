'''Uses qsub to launch iskay_AperturePhotomertyPreproc.py which
applies aperture photometry to current dataset.

How many cores we request and use is governed by the environment variable
OMP_NUM_THREADS, make sure you have it set to something sensical.
In my tests, a values of OMP_NUM_THREADS=2 works fine.

Written by P. Gallardo.'''
from iskay import paramTools
from iskay import catalogTools
from iskay import envVars
import numpy as np
import os
#import time
import shutil
import datetime


def howManyEntries():
    '''Opens the catalog pointed to in params.ini and returns its length
    are there.'''
    params = paramTools.params('params.ini')
    df = catalogTools.cat(fname=params.CAT_FNAME).df
    return len(df)


def qsubCommand(chunkSize, chunkNumber):
    '''Gets the chunk size and the chunk number to proces, generates
    the command to launch qsub'''
    N_threads = int(os.environ['OMP_NUM_THREADS']) + 1
    T = datetime.datetime.now()
    dt = datetime.timedelta(seconds=10*chunkNumber)
    T = T+dt
    cmd = []
    cmd.append("qsub -b y -q all.q -S /bin/bash -l mem_free=30G")
    cmd.append('-pe sge_pe %i -l m_core=%i' % (N_threads, N_threads))
#    cmd.append('-o /dev/null')
#    cmd.append('-e /dev/null')
    cmd.append("-a %s" % T.strftime('%Y%m%d%H%M.%S'))
    cmd.append("-o /tmp/pag227_sge_%i.out" % chunkNumber)
    cmd.append("-e /tmp/pag227_sge_%i.err" % chunkNumber)
    cmd.append("-cwd -V -l")
    cmd.append("h_rt=48:00:00")
    cmd.append("/nfs/user/pag227/miniconda/bin/python")
    cmd.append("/home/pag227/code/iskay/misc/"
               "iskay_AperturePhotomertyPreprocOneChunk.py")
    cmd.append("%i %i" % (chunkSize, chunkNumber))
    return " ".join(cmd)


if os.path.exists('ApPhotoResults'):
    shutil.rmtree('ApPhotoResults')  # clean results folder

howMany = howManyEntries()
RUNLOCALLY = False
#MAX_WAIT = 20  # seconds to wait, so we don't swamp the dispatcher.
if RUNLOCALLY:
    print("We need to implement the local executor.")
    assert False
    cmd = ['python']
    cmd.append("/home/pag227/code/iskay/misc/"
               "iskay_AperturePhotomertyPreprocOneChunk.py")
    cmd.append("%i 0" % envVars.ApPhotoMaxLengthToProcessLocally)
else:
    chunkSize = envVars.ApPhotoMaxLengthToProcessLocally
    nit = int(np.ceil(float(howMany)/float(chunkSize)))
    for chunkNumber in range(nit):
        cmd = qsubCommand(chunkSize, chunkNumber)
        print cmd
        os.system(cmd)
        print('Job %i submitted...' % chunkNumber)

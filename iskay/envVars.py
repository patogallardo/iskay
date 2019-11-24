'''Envirnoment variables that point to where the data lives.
And where the local folder to store it is.'''
import os
#Environment variables go here.
#Define folders where things run and are stored. The rest of the
#code uses this to know where to point to.

#this is the local folder where data are. We will create it if it
#doesn't exist:
localDataPath = '/tmp/pag227/data/iskay/'

#These are the folders where the data comes from:
#If data does not exist in the local folder, then it will be transfered
#from the Remote Path.
remotePath1 = '/nfs/grp/cosmo/kSZ/'  #primary source of data
remotePath2 = '/nfs/grp/cosmo/Pato/kSZ/data'  #secondary source of data

if not os.path.exists(localDataPath):
    print "localDataPath does not exist, creating it."
    try:
        os.makedirs(localDataPath)
    except:
        pass  # this is needed for running on a cluster


assert os.path.exists(remotePath1)
assert os.path.exists(remotePath2)

# Aperture photometry preprocessing. In how many parts do we split
# the calculation? Increasing this number reduces the wait time for this
# step, but going too high will give diminishing returns.
ApPhotoMaxLengthToProcessLocally = 70000

#Parameters for distributed computations
Ncores = 20  #how many cpu cores will have each worker
NWorkers = 11  # how many workers do we want running at the same time

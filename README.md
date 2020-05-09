# iskay
## an estimator of the pairwise kinetic sunyaev-zeldovich effect.

![alt text](https://raw.githubusercontent.com/patogallardo/iskay/master/imgs/cover.PNG "ksz diagram")

Iskai is a tool that computes the pairwise kinetic Sunyaev-Zeldovich effect estimator starting from maps of the CMB. It is able to use a catalog of galaxy positions, redshifts and luminosities with a map of the CMB and returns an estimator that minimizes the mean square error (MSE).

This code is able to: take a catalog of galaxy positions, redshifts and luminosities, produce aperture photometry for the given galaxy centerings with a circular aperture  which does not distort the frequency domain content of the substamps (via zero-padding), reprojects to a locally square grid and naturally takes care of the sub-pixel position of galaxy centerings by finely discretizing the interpolated map. The reprojection and interpolation is done nativelly in pixell.

The computation is split into two parts. The aperture photometry extraction and the ksz estimatimation.

1) The aperture photometry extraction uses the pixell library, which takes care of the reprojection and supersampling to compute the fractional pixel weighting, which allows proper centering. This part of the computation is embarrasingly parallel, so the computation is split into segments and iskay handles each chunk separately in a slurm type cluster.

2) The pairwise ksz estimation computes the geometrical weighhts c_ij and carries out the sum over all the pairs of objects for a given luminosity cut (actually, the user can sepecify an arbitrary cut in any of the variables the catalog contains, these are passed in the form of a text string in a format compliant with the pandas dataframe.query method). 

### kSZ curves

The kSZ curve can be computed with equal weights or variance weights. The variance weighted estimator is computed following equation 5.26 in [1], while the equal weights version is more standard and is presented in [2] and the references therein.

### Jackknifes
For a given ksz curve, the estimation of errorbars is implemented by the use of a jackknife resampling in galaxy space (as opposed to galaxy pair space which is not free from assumptions). Computation of covariance matrices from this resampling is supported. 

### Implementation details
This software is designed to be deployed on a cluster. Fast implementation was achieved in Python using a just in time compiler (numba), which enables the use of multiple cores per machine. Massive deployments are supported by the use of Dask. This is done to run all the jackknife replicants in different nodes in a supercomputer.

Written by P. Gallardo based on code named "pairwiser" by F. de Bernardis.

## Usage

All the user accesible parameters are controlled in a params.ini file. This file can be generated using misc/iskay_makeParamFile.py. This file contains: the path to the fits file, the number of bins and bin separation (uneven bins are also supported), the path to the catalog, the query, aperture photometry disk and ring sizes in arcminutes and other parameteres.

### Aperture photometry
With the params.ini file in place, the aperture photometry can be launched with the script 'iskay_preprocessAperturePhotometry.py' which will split the catalog into chunks and run the aperture photometry on each one of the subgroups. The output will be a bunch of catalog files stored in the ApPhotoResults folder, which will contain one csv file per galaxy segment. Note that this part of the computation is split into pieces just to make the computation faster. The computation of each aperture photometry sub-catalog is done in the script 'misc/iskay_AperturePhotomertyPreprocOneChunk.py' which is called iteratively by 'iskay_preprocessAperturePhotometry.py'.

### Pairwise calculation, jackknifes and covariance matrices

This is the heavy part of the calculation. It was implemented using Dask to split the calculation into parts and run all the parts for the jackknife calculation. The script that performs the jackknife calculation is located in 'misc/iskay_analysis.py'. This script will:

1) Open the aperture photometry data stored in ApPhotoResults
2) Compute the ksz curve for the whole dataset. This is done in parallel in one machine with multiple cores.
3) Divide the catalog into pieces (jackknife replicants) and run the estimator on each one of the replicants. Each of the resampled catalogs (replicats) go to a different instance of the ksz curve estimator. The estimator is run using the same function that was run for the whole catalog.

### Results

Results are stored in a pickle object. The pickle object is called a 'JK' which has all the variables of interest. Like: the ksz curve, the separation bins, all the jk replicants, the covariance matrix, the errorbars. For more detail on what variables are stored, see iskay/JK.py and iskay/JK_tools.py.

## Tests

Effort has been put into testing this pipeline. The folder tests contains automated tests that are done to ensure consistency. The method I used was the following: Run FdB's pairwise estimator on an aperture photometry sample, write a single threaded version for iskay. Parallelize the pairwiser function while ensuring equal results to the parent functions in FdB's code. Once the parallel version is implemented, implement the distribted version and compare results to the parent functions to ensure no data is being missed.

Modifications to the original code were made such that: variance weighting reproduces the original FdB's code for equal weights, uneven binning tools also are able to generate the original results for equal width bins.

## References:
[1] Gallardo 2019: Optimizing future CMB observatories and measuring galaxy cluster motions with the Atacama Cosmology Telescope

[2] deBernardis 2016: Detection of the pairwise kinematic Sunyaev-Zel'dovich effect with BOSS DR11 and the Atacama Cosmology Telescope

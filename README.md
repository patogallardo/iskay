# iskay: 
## an estimator of the pairwise kinetic sunyaev-zeldovich effect.

![alt text](https://raw.githubusercontent.com/patogallardo/iskay/master/imgs/cover.PNG "ksz diagram")

Computes the pairwise kinetic Sunyaev-Zeldovich effect estimator starting from maps of the CMB.

This code is able to: take a catalog of galaxy positions, redshifts and luminosities, produce aperture photometry for the given galaxy centerings with a circular aperture  which does not distort the frequency domain content of the substamps (via zero-padding), reprojects to locally square grid and naturally takes care of sub pixel position of galaxy centerings by finely discretizing the interpolated map. The reprojection and interpolation is done nativelly in pixell.

The computation is split into two parts. The aperture photometry extraction and the ksz estimatimation.

The aperture photometry extraction uses the pixell library, which takes care of the reprojection and supersampling to compute the fractional pixel weighting, which allows proper centering. This part of the computation is embarrasingly parallel, so the computation is split into segments and iskay handles each chunk separately in a slurm type cluster.

The pairwise ksz estimation computes the geometrical weighhts c_ij and carries out the sum over all the pairs of objects for a given luminosity cut (actually, the user can sepecify an arbitrary cut in any of the variables the catalog contains, these are passed in the form of a text string in a format compliant with the pandas dataframe.query method). 

The kSZ curve can be computed with equal weights or variance weights. The variance weighted estimator is computed following equation 5.26 in [1], while the equal weights version is more standard and is presented in [2] and the references therein.

For a given ksz curve, the estimation of errorbars  implemented by the use of a jackknife resampling in galaxy space (as opposed to galaxy pair space which is not free from assumptions). Computation of covariance matrices from this resampling is supported. 

This piece of software is designed to be deployed on a cluster. Fast implementation was achieved in Python using a just in time compiler (numba), which enables the use of multiple cores per machine. Massive deployments are supported by the use of Dask and this is done to run all the jackknife replicants in different nodes in a supercomputer.


Written by P. Gallardo based on code named "pairwiser" by F. de Bernardis.

## References:
[1] Gallardo 2019: Optimizing future CMB observatories and measuring galaxy cluster motions with the Atacama Cosmology Telescope

[2] deBernardis 2016: Detection of the pairwise kinematic Sunyaev-Zel'dovich effect with BOSS DR11 and the Atacama Cosmology Telescope



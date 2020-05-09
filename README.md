# iskay: Estimator of the pairwise kinetic sunyaev-zeldovich effect.

V1: Written by P. Gallardo based on code named "pairwiser" by F. de Bernardis.

Computes the pairwise estimator starting from maps of the CMB.

The code is able to: take a catalog of galaxy positions, redshifts and luminosities, produce aperture photometry for the given galaxy centerings with a circular aperture  which does not distort the frequency domain content of the substamps, reprojects to locally square grid and naturally takes care of sub pixel position of galaxy centerings by finely discretizing the interpolated map. The reprojection and interpolation is done nativelly in pixell.

This piece of software is designed to be deployed on a cluster. Fast implementation was achieved in Python using a just in time compiler (numba), which enables the use of multiple cores per machine. Massive deployments are supported by the use of Dask.

The computation is split into two parts. The aperture photometry extraction and the ksz estimation.

The ksz estimation can be done with equal weights and variance weights. The variance weighted estimator is computed following equation 5.26 in [1].

For a given ksz curve, the estimation of errorbars is supported by the use of a jackknife in galaxy space (as opposed to galaxy pair space which is not free from assumptions). The jackknife 

[1] Gallardo 2019 Optimizing future CMB observatories and measuring galaxy cluster motions with the Atacama Cosmology Telescope

#!/bin/bash
#$ -S /bin/bash
#$ -l mem_free=30G
#$ -l hostname=lnx1032-f1
#$ -pe sge_pe 15
#$ -l m_core=15
#$ -q all.q
#$ -o /home/pag227/sge_logs/sge_exportSubmaps.out
#$ -e /home/pag227/sge_logs/sge_exportSubmaps.err
#$ -cwd
export MKL_NUM_THREADS=14
export OMP_NUM_THREADS=14
rm /tmp/pag227/ApPhotoResults/*.h5
/nfs/user/pag227/miniconda/bin/python /home/pag227/code/iskay/misc/iskay_exportSubmaps.py submaps
mv /tmp/pag227/ApPhotoResults/*.h5 /nfs/grp/cosmo/kSZ/V20_DR15Catalog_submaps/

/nfs/user/pag227/miniconda/bin/python /home/pag227/code/iskay/misc/iskay_exportSubmaps.py divmaps
mv /tmp/pag227/ApPhotoResults/*.h5 /nfs/grp/cosmo/kSZ/V20_DR15Catalog_submaps/

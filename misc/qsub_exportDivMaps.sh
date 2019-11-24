#!/bin/bash
#$ -S /bin/bash
#$ -l mem_free=30G
#$ -l hostname=lnx1032-f1
#$ -pe sge_pe 13 
#$ -l m_core=13
#$ -q all.q
#$ -o /home/pag227/sge_logs/sge_exportSubmaps.out
#$ -e /home/pag227/sge_logs/sge_exportSubmaps.err
#$ -cwd
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12
/nfs/user/pag227/miniconda/bin/python /home/pag227/code/iskay/misc/iskay_exportSubmaps.py divmaps

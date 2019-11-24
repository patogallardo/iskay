#~!/bin/bash
#$ -q all.q
#$ -S /bin/bash
#$ -l mem_free=8G
#$ -o /home/pag227/pairwiserSandBoxes/sge_output/sge.out
#$ -e /home/pag227/pairwiserSandBoxes/sge_output/sge.err
#$ -pe sge_pe 11
#$ -l m_core=11
#$ -cwd
export NUMBA_NUM_THREADS=10

/nfs/user/pag227/miniconda/bin/python exportTzav.py

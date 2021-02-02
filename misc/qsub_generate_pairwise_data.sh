#!/bin/bash
#$ -S /bin/bash
#$ -l mem_free=10G
#$ -pe sge_pe 2
#$ -l m_core=2
#$ -q all.q
#$ -o /home/pag227/sge_logs/sge_pairwise.out
#$ -e /home/pag227/sge_logs/sge_pairwise.err
#$ -t 1-40
#$ -cwd

ssh lnx1032-f1 rm /tmp/pag227/pairwise_data/*.hdf

/nfs/user/pag227/miniconda/bin/python ~/code/iskay/misc/iskay_compute_all_pairs_and_save_one_chunk.py 40 $SGE_TASK_ID

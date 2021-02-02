#!/nfs/user/pag227/miniconda/bin/python
import glob
import os

fnames = glob.glob('params_*.ini')
fnames.sort()
jobid_past = None

for fname in fnames:
    print(fname)

#for j in range(len(fnames)):
for j in range(len(fnames)):
    cmd = []
    jobid = 'chainjob%02i' % (j)
    cmd.append('qsub -b y -q all.q -o /home/pag227/sge_logs/sge_%i.out' % j)
    cmd.append('-e /home/pag227/sge_logs/sge_%i.err' % j)
    cmd.append('-cwd')
    cmd.append('-N %s' % jobid)
    if j > 0:
        cmd.append('-hold_jid %s' % jobid_past)
    cmd.append('/nfs/user/pag227/miniconda/bin/python')
    cmd.append('./iskay_analysis.py')
    cmd.append(fnames[j])
    cmd = " ".join(cmd)
    jobid_past = jobid
    print cmd

    os.system(cmd)

cmd = []
jobid = 'export_data'
cmd.append('qsub -b y -q all.q -o /home/pag227/sge_logs/sge_%i.out' % j)
cmd.append('-e /home/pag227/sge_logs/sge_%i.err' % j)
cmd.append('-cwd')
cmd.append('-N %s' % jobid)
cmd.append('-hold_jid %s' % jobid_past)
cmd.append('/nfs/user/pag227/miniconda/bin/python')
cmd.append('/home/pag227/code/iskay/misc/iskay_export_cov_mat.py')
cmd = " ".join(cmd)
jobid_past = jobid
print cmd
os.system(cmd)

cmd = []
jobid = 'export_covmat_data'
cmd.append('qsub -b y -q all.q -o /home/pag227/sge_logs/sge_%i.out' % j)
cmd.append('-e /home/pag227/sge_logs/sge_%i.err' % j)
cmd.append('-cwd')
cmd.append('-N %s' % jobid)
cmd.append('-hold_jid %s' % jobid_past)
cmd.append('/nfs/user/pag227/miniconda/bin/python')
cmd.append('/home/pag227/code/iskay/misc/iskay_export_cov_for_fits.py')
cmd = " ".join(cmd)
jobid_past = jobid
print cmd
os.system(cmd)

cmd = []
jobid = 'plot_curves'
cmd.append('qsub -b y -q all.q -o /home/pag227/sge_logs/sge_%i.out' % j)
cmd.append('-e /home/pag227/sge_logs/sge_%i.err' % j)
cmd.append('-cwd')
cmd.append('-N %s' % jobid)
cmd.append('-hold_jid %s' % jobid_past)
cmd.append('/nfs/user/pag227/miniconda/bin/python')
cmd.append('/home/pag227/code/iskay/misc/iskay_make_ksz_curve_plots.py')
cmd = " ".join(cmd)
jobid_past = jobid
print cmd
os.system(cmd)


cmd = []
jobid = 'plot_covmats'
cmd.append('qsub -b y -q all.q -o /home/pag227/sge_logs/sge_%i.out' % j)
cmd.append('-e /home/pag227/sge_logs/sge_%i.err' % j)
cmd.append('-cwd')
cmd.append('-N %s' % jobid)
cmd.append('-hold_jid %s' % jobid_past)
cmd.append('/nfs/user/pag227/miniconda/bin/python')
cmd.append('/home/pag227/code/iskay/misc/iskay_makeCorrMatPlots.py')
cmd = " ".join(cmd)
jobid_past = jobid
print cmd
os.system(cmd)

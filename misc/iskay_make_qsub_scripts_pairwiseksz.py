#!/nfs/user/pag227/miniconda/bin/python
import glob
import sys

assert len(sys.argv) == 2  # usage make_qusb...py "params_*_bs_dt.ini"

if len(sys.argv) == 2:
    fname_wildcard = sys.argv[1]
else:
    fname_wildcard = "params_*.ini"
fnames = glob.glob(fname_wildcard)
fnames.sort()

print("Making qsub file for param files:")
for fname in fnames:
    print(fname)

line1 = ("qsub "
         " -b y -cwd -N run00 /nfs/user/pag227/miniconda/bin/python "
         "./iskay_analysis.py %s" % fnames[0])

lines = []
lines.append(line1)

for j in range(1, len(fnames)):
    line = ('qsub '
            ' -b y -cwd -N run%02i -hold_jid run%02i '
            '/nfs/user/pag227/miniconda/bin/python'
            ' ./iskay_analysis.py %s' % (j, j-1, fnames[j]))
    lines.append(line)

tofile = "\n".join(lines)

with open("qsub_launch_pairwise.sh", 'w') as f:
    f.write(tofile)

#!/nfs/user/pag227/miniconda/bin/python
from iskay import paramTools
import glob

p = paramTools.params('./params.ini')

TITLE = p.NAME

s1 = []
s1.append('# %s\n\n' % p.NAME)
s1.append("**fits_fname** = %s \n\n" % p.FITS_FNAME)
s1.append("**divmap_fname** = %s \n\n" % p.DIVMAP_FNAME)
s1.append("**query** = %s\n\n" % p.CAT_QUERY)

s1.append("# Curves\n")

s1.append("![](kSZ_velocityCurves.png)\n")
s1.append("![](kSZ_velocityCurves_disjoint_bins.png)\n")

s1.append("# Covariance Matrices\n")


def get_files(token):
    '''Find files under folder token'''
    s = []
    fnames = glob.glob("./%s/%s_lum*.png" % (token, p.NAME.replace('.', 'p')))

    s.append("## Joint bins\n")

    for fname in fnames:
        fname = fname.split("/")[-1]
        s.append("![](%s)\n" % fname)

    fnames = glob.glob("./%s/%s_disjoint*.png" % (token,
                                                  p.NAME.replace(".", 'p')))

    s.append("## Disjoint bins\n")
    for fname in fnames:
        fname = fname.split("/")[-1]
        s.append("![](%s)\n" % fname)
    return s


s_jk = s1 + get_files('results_jk')
tofile = "".join(s_jk)

with open("./results_jk/readme.md", 'w') as f:
    f.write(tofile)

s_bs = s1 + get_files('results_bsdt')
tofile = "".join(s_bs)

with open('./results_bsdt/readme.md', 'w') as f:
    f.write(tofile)

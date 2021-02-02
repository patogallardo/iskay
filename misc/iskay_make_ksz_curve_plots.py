#!/nfs/user/pag227/miniconda/bin/python
import matplotlib
matplotlib.use('agg')

from iskay import JK  # noqa
import matplotlib.pyplot as plt  # noqa
import glob  # noqa
import re  # noqa

fnames_jk = glob.glob('./results_jk/*_lum_gt_??p?_jk.pck')
fnames_jk.sort()
fnames_bs = glob.glob('./results_bs/*_lum_gt_??p?_bootstrap.pck')
fnames_bs.sort()
fnames_bsdt = glob.glob('./results_bsdt/*_lum_gt_??p?_bs_dt.pck')
fnames_bsdt.sort()


def make_plots(fnames):
    if len(fnames) == 0:
        pass
    else:
        print("Making plots for:")
        plt.figure(figsize=[8, 4.5])
        for j in range(len(fnames)):
            fname = fnames[j]
            print(fname)
            jk = JK.load_JK(fname)
            m = re.match("(lum.*e10)", jk.query)
            label = m.group(1)
            label = label + ', N: %1.1fk' % (
                             jk.N_objects_in_this_run/1e3)  # noqa
            plt.errorbar(jk.rsep + j*1.5, jk.kSZ_curveFullDataset,
                         yerr=jk.errorbars,
                         label=label, ls='', marker='.')
        if 'jk' in fname:
            folder = 'results_jk'
        elif 'bs' in fname:
            if 'bsdt' in fname:
                folder = 'results_bsdt'
            else:
                folder = 'results_bs'
        plt.axhline(0, color='black')
        plt.legend(loc='lower right')
        plt.title(jk.name[:jk.name.find('_lum_')])
        plt.xlabel('$r_{sep} [Mpc]$')
        plt.ylabel('p [$\\mu K$]')
        if 'disjoint' in fname:
            fname_out = '%s/kSZ_velocityCurves_disjoint_bins' % (folder)
        else:
            fname_out = '%s/kSZ_velocityCurves' % (folder)
        plt.ylim([-0.3, 0.3])
        plt.savefig(fname_out + '.png', dpi=120)
        plt.savefig(fname_out + '.pdf')
        #plt.show()
        plt.close()


make_plots(fnames_jk)
make_plots(fnames_bs)
make_plots(fnames_bsdt)

fnames_jk = glob.glob('./results_jk/*_disjoint_*jk.pck')
fnames_jk.sort()
fnames_bs = glob.glob('./results_bs/*_disjoint_*bootstrap.pck')
fnames_bs.sort()
fnames_bsdt = glob.glob('./results_bsdt/*_disjoint*.pck')
fnames_bsdt.sort()

make_plots(fnames_jk)
make_plots(fnames_bs)
make_plots(fnames_bsdt)

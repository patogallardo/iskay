import matplotlib.pyplot as plt
from iskay import JK
import glob
import numpy as np
import os


def plot_all_kSZ_curves_JKs_in_current_folder(show=True):
    '''In the current folder, it looks for all files with .jk extension.
    For all those files, plots the velocity curve with JK errorbars.

    show: shows the plot interactively, False: saves the figure to disk.'''
    fnames = glob.glob('*.pck')
    for j, fname in enumerate(fnames):
        jk = JK.load_JK(fname)
        label = '%s (%i gals)' % (jk.query, jk.N_objects_in_this_run)
        plt.errorbar(jk.rsep + j,
                     jk.kSZ_curveFullDataset,
                     yerr=jk.errorbars,
                     marker='o', linestyle='',
                     label=label)
    plt.xlabel('$r_{sep}$[Mpc]')
    plt.ylabel('p [$\\mu$K]')
    plt.axhline(0, color='black')
    plt.legend(loc='upper right')
    if show:
        plt.show()
    else:
        plt.savefig('kSZ_curves.pdf')
        plt.savefig('kSZ_curves.png', dpi=200)
        plt.close()


def plot_all_JK_errorbars_in_current_folder(show=True):
    '''Idem to plot_all_kSZ_curves_JKs_in_current_folder but for error bar
    only plots.'''
    fnames = glob.glob('*.pck')
    for fname in fnames:
        jk = JK.load_JK(fname)
        label = '%s (%i gals)' % (jk.query, jk.N_objects_in_this_run)
        plt.scatter(jk.rsep, jk.errorbars, marker='o', label=label)
    plt.xlabel('$r_{sep}$[Mpc]')
    plt.ylabel('p [$\\mu$K]')
    plt.legend(loc='upper right')
    plt.title('JK (%i it.) errorbars' % jk.JK_Ngroups)
    if show:
        plt.show()
    else:
        plt.savefig('JK_errorbars.pdf')
        plt.savefig('JK_errorbars.png', dpi=200)
        plt.close()


def plot_JK_runtimes_in_current_folder(show=True):
    '''In the current folder, plot the run times for all the .pck files.'''
    fnames = glob.glob('*.pck')
    runtimes = []
    n_objects = []
    for fname in fnames:
        jk = JK.load_JK(fname)
        runtimes.append(jk.runtime)
        n_objects.append(jk.N_objects_in_this_run)
    runtimes = np.array(runtimes)
    n_objects = np.array(n_objects)

    argsorted = np.argsort(n_objects)
    n_objects = n_objects[argsorted]
    runtimes = runtimes[argsorted]
    plt.scatter(n_objects, runtimes)
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    plt.title('Distributed iskay 50JK performance\n'
              '20 cores per machine, 10 machines')
    plt.xlabel('N$_{galaxies}$')
    plt.ylabel('Runtime [s]')
    if show:
        plt.show()
    else:
        if not os.path.exists('plots'):
            os.makedirs('plots')
        plt.savefig('plots/runtimes.pdf')
        plt.close()

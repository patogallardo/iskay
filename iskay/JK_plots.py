'''Amenities for plotting jk objects.

Written by P. Gallardo
'''
import matplotlib.pyplot as plt
import seaborn as sns


def plot_kSZ_curveWithErrorbars(jk, show=True):
    '''Receives a jk object. Makes an errorbar plot of the velocity curve.
    draws a line around zero.'''
    plt.errorbar(jk.rsep, jk.kSZ_curveFullDataset, jk.errorbars,
                 label=jk.query)
    plt.legend(loc='upper right')
    plt.xlabel('r [Mpc]')
    plt.ylabel('p [$\\mu$K]')
    plt.axhline(0, color='black')
    if show:
        plt.show()
    else:
        plt.savefig(jk.name + '.png', dpi=200)
        plt.savefig(jk.name + '.pdf')
        plt.close()


def plotCorrMatrix(jk, show=True):
    '''Receives a jk object. plots correlation matrix.
    '''
    sns.heatmap(jk.corr, vmin=-0.2, vmax=1)
    title = 'Corr matrix %s, %s' % (jk.query, jk.name)
    if len(title) > 20:
        title = 'Corr matrix %s' % (jk.name)  # patch for long title
    plt.title(title)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.savefig('%s_corrMatrix.png' % jk.name, dpi=200)
        plt.close()

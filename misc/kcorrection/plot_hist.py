'''Makes histogram of the luminosity column in the csv file.
'''
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

show = False

fname = sys.argv[1]
assert os.path.exists(fname)  # path must exist

df = pd.read_csv(fname)

# first plot
plt.figure(figsize=[8, 4.5])
plt.hist(df.lum, range=[0, 2e11], bins=1000, histtype='step', density=True,
         color='black')

plt.xlabel('lum [$L_\\odot$]')
plt.ylabel('pdf [$L_\\odot^{-1}$]')
if show:
    plt.show()
else:
    plt.savefig('./output/lum_pdf.png', dpi=120)
    plt.close()

# second plot
plt.figure(figsize=[8, 4.5])
plt.hist(df.lum, range=[0, 2e11], bins=1000, histtype='step', density=True,
         cumulative=True, color='black')

plt.xlabel('lum [$L_\\odot$]')
plt.ylabel('cdf')
plt.yticks(np.arange(0, 1.10, 0.10))
plt.xticks(np.arange(0, 2.1e11, 0.1e11))
plt.grid()

if show:
    plt.show()
else:
    plt.savefig('./output/lum_cdf.png', dpi=120)
    plt.close()

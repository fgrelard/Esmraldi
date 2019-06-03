import src.spectraprocessing as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os



#spectra = np.load("data/spectra.npy")
# x0 = np.array(spectra[0][0])
# y0 = np.array(spectra[0][1])

indices = np.load("data/indices.npy")
indices2 = sp.peak_reference_indices(indices)
print(indices2)
indices = np.unique(indices)
print(indices[:50:])

print(len(indices), len(indices2))
spectra_m = sp.spectra_mean(spectra)

mpl.rcParams["savefig.directory"] = os.chdir("/mnt/c/Users/fgrelard/Documents/MALDI/")

plt.plot(x0, spectra_m, x0[indices2], spectra_m[indices2], ".")
plt.show()

import src.spectraprocessing as sp
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os



spectra = np.load("data/spectra_small.npy")
spectra_mean_before = sp.spectra_sum(spectra)
realigned_spectra = sp.realign(spectra, 50)
spectra_mean_after = sp.spectra_sum(realigned_spectra)
# unique_indices = np.unique(sp.spectra_peak_indices(spectra))

# unique_spectra = []
# for spectrum in spectra:
#     x,y = spectrum
#     unique_spectra.append((x[unique_indices], y[unique_indices]))

# for i in range(len(spectra)):
#     plt.plot(unique_spectra[i][0], unique_spectra[i][1], ".",realigned_spectra[i][0], realigned_spectra[i][1], "o")
#     plt.show()

plt.plot(spectra[0][0], spectra_mean_before, realigned_spectra[0][0], spectra_mean_after, ".")
plt.show()

import src.spectraprocessing as sp
import src.imzmlio as io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os



spectra = np.load("data/peaksel_3.npy")
prominence = 50
spectra_max_before = sp.spectra_max(spectra)
#spectra_max_after = sp.spectra_max(realigned_spectra)
indices_spectra_max = sp.peak_indices(spectra_max_before, prominence)

realigned_spectra = sp.realign(spectra, prominence)
#p = io.open_imzml("/mnt/d/MALDI/imzML/MSI_20190419_01/00/00.imzML")

mzs = []
intensities = []
to_array = []
for spectrum in realigned_spectra:
    x, y = spectrum
    mzs.append(x)
    intensities.append(y)

#np.save("data/peaksel_2.npy", np.asarray(to_array))

io.write_imzml(mzs, intensities, p.coordinates, "/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel_2.imzML")


full_indices = sp.spectra_peak_indices(spectra, prominence)
small_indices = sp.peak_reference_indices(full_indices)

print(len(small_indices), " ", len(indices_spectra_max))

print(len(realigned_spectra[0][0]))
plt.plot(spectra[0][0], spectra_max_before, realigned_spectra[0][0], np.array(spectra_max_before)[small_indices], ".")
plt.show()

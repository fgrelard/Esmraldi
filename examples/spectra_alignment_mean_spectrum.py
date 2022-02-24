import argparse
import os
import numpy as np
import matplotlib.pyplot as plt

import esmraldi.spectraprocessing as sp
import esmraldi.imzmlio as io
from esmraldi.peakdetectionmeanspectrum import PeakDetectionMeanSpectrum


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-p", "--prominence", help="Prominence factor")
parser.add_argument("-s", "--step", help="Step ppm")
parser.add_argument("-m", "--mz", help="M/Z ion image to display after realignment", default=0)
args = parser.parse_args()

input_name = args.input
prominence = float(args.prominence)
step = float(args.step)
mz_reference = float(args.mz)

imzml = io.open_imzml(input_name)
max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
max_y = max(imzml.coordinates, key=lambda item:item[1])[1]

npy_name = os.path.splitext(input_name)[0] + "_spectra.npy"
if os.path.isfile(npy_name):
    print("Loading")
    spectra = np.load(npy_name, allow_pickle=True)
else:
    spectra = io.get_spectra(imzml)
    np.save(npy_name, spectra)

print("Mean spectrum")
npy_name = os.path.splitext(input_name)[0] + "_meanspectra.npy"
if os.path.isfile(npy_name):
    mzs, mean_spectrum = np.load(npy_name)
else:
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
    mean_spectrum = sp.spectra_mean_centroided(spectra, mzs)
    np.save(npy_name, [mzs, mean_spectrum])

print("Peak detection")
peak_detection = PeakDetectionMeanSpectrum(mzs, mean_spectrum, prominence, step)
peak_indices = peak_detection.extract_peaks()
peaks = mzs[peak_indices]
intensities = mean_spectrum[peak_indices]

plt.plot(mzs, mean_spectrum)
plt.plot(peaks, intensities, "o")
plt.show()

realigned_spectra = sp.realign_generic(spectra, peaks)
mz_index = np.abs(peaks - mz_reference).argmin()

full_spectra_sparse = io.get_full_spectra_sparse(realigned_spectra, max_x*max_y)
image = io.get_images_from_spectra(full_spectra_sparse, (max_x, max_y))

mz_index = np.abs(peaks - mz_reference).argmin()

plt.imshow(image[..., mz_index].T)
plt.show()

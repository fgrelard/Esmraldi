import src.spectraprocessing as sp
import src.imzmlio as io
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import argparse


def plot_peak_selected(spectra, realigned_spectra):
    spectra_max_before = sp.spectra_max(spectra)
    #spectra_max_after = sp.spectra_max(realigned_spectra)
    indices_spectra_max = sp.peak_indices(spectra_max_before, prominence)

    full_indices = sp.spectra_peak_indices(spectra, prominence)
    small_indices = sp.peak_reference_indices_groups(full_indices)

    print(len(small_indices), " ", len(indices_spectra_max))

    print(len(realigned_spectra[0][0]))
    plt.plot(spectra[0][0], spectra_max_before, realigned_spectra[0][0], np.array(spectra_max_before)[small_indices], ".")
    plt.show()

def extract_mz_above(spectra):
    indices = spectra[0,0,...] > 1700
    smaller_spectra = spectra[..., indices]
    return smaller_spectra


# spectra = np.load("data/old/peaksel_2.npy")
# deisotoped_spectra = sp.deisotoping(spectra)
# np.save("data/peaksel_2_deisotoped.npy", deisotoped_spectra)
# print(spectra.shape)
# exit(0)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI imzML")
parser.add_argument("-o", "--output", help="Output peak selected imzML")
args = parser.parse_args()

inputname = args.input
outname = args.output

p = io.open_imzml(inputname)

spectra = io.get_spectra(p)

prominence = 75

print("Realignment")
realigned_spectra = sp.realign(spectra, prominence)

print("Deisotoping")
averagine = {'C': 7.0, 'H': 11.8333, 'N': 0.5, 'O': 5.16666}
deisotoped_spectra = sp.deisotoping_simple(realigned_spectra, nb_charges=2, average_distribution=averagine)

# deisotoped_spectra = sp.deisotoping(np.array(realigned_spectra))

print(realigned_spectra.shape)
print(deisotoped_spectra.shape)
print(deisotoped_spectra[0, 0, ...])

mzs = []
intensities = []
to_array = []
for spectrum in deisotoped_spectra:
    x, y = spectrum
    mzs.append(x)
    intensities.append(y)

#np.save("data/peaksel_250DJ_prominence75.npy", realigned_spectra)

io.write_imzml(mzs, intensities, p.coordinates, outname)

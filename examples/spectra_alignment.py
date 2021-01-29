"""
MALDI processing:
  1/ Peak selection
  2/ Realignment
  3/ Deisotoping
"""
import os
import math
import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import esmraldi.spectraprocessing as sp
import esmraldi.imzmlio as io
import esmraldi.speciesrule as sr
import esmraldi.spectrainterpretation as si
from esmraldi.theoreticalspectrum import TheoreticalSpectrum
import sys

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



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI imzML")
parser.add_argument("-o", "--output", help="Output peak selected imzML")
parser.add_argument("-p", "--prominence", help="Prominence for peak selection", default=75)
parser.add_argument("-n", "--nbpeaks", help="Number of occurrence of peaks across all the spectra for the realignment", default=4)
parser.add_argument("-z", "--nbcharges", help="Number of charges for deisotoping", default=2)
parser.add_argument("-s", "--step", help="Tolerance step to realign peaks (in m/z)", default=0.05)
parser.add_argument("-t", "--tolerance", help="Tolerance for deisotoping (in m/z)", default=0.05)
parser.add_argument("-d", "--deisotope", help="Whether the realigned spectra should be deisotoped", action="store_true")
parser.add_argument("--normalize", help="TIC normalization")
parser.add_argument("--theoretical", help="If provided, only peaks close to the theoretical spectrum are kept")
parser.add_argument("--tolerance_theoretical", help="Tolerance to match two peaks between theoretical spectrum and observed spectrum", default=0.1)
parser.add_argument("--no_picking", help="Whether to perform no picking and alignement. Only deisotoping is done.", action="store_true")
parser.add_argument("--noise_level", help="Define noise level manually. If not provided, assumed as the standard deviation of the minimum spectrum.", default=0)
args = parser.parse_args()

inputname = args.input
outname = args.output
prominence = int(args.prominence)
nb_peaks = int(args.nbpeaks)
nb_charges = int(args.nbcharges)
step_mz = float(args.step)
tolerance_mz = float(args.tolerance)
is_deisotoped = args.deisotope
is_normalized = args.normalize
theoretical = args.theoretical
tolerance_theoretical = float(args.tolerance_theoretical)
is_nopicking = args.no_picking
noise_level = float(args.noise_level)

np.set_printoptions(threshold=sys.maxsize)

p = io.open_imzml(inputname)
spectra = io.get_spectra(p)

mz, I = spectra[0]

print("Realignment")
min_diff = mz[1] - mz[0]
wlen = max(int(len(mz)/10), int(50.0 / min_diff))

print("Spectral resolution= ", min_diff)
print("Window length= ", wlen)


realigned_spectra = spectra
if not is_nopicking:
    if noise_level <= 0:
        peak_indices = sp.spectra_peak_indices_adaptative(spectra, factor=prominence, wlen=wlen)
    else:
        peak_indices = sp.spectra_peak_indices_adaptative_noiselevel(spectra, factor=prominence, wlen=wlen, noise_level=noise_level)
    mzs = np.array([x[peak_indices[i]] for i, (x,y) in enumerate(spectra)], dtype=object)

    spectra = sp.normalization_sic(spectra, peak_indices)

    # realigned_spectra = sp.realign(spectra, prominence, nb_peaks)
    realigned_spectra = sp.realign_mzs(spectra, mzs, reference="median", nb_occurrence=nb_peaks, step=step_mz)
    print(realigned_spectra.shape)

print("Before deisotoping", realigned_spectra[0, 0, ...])
print("Length =", len(realigned_spectra[0, 0, ...]))
if is_deisotoped:
    print("Deisotoping")
    averagine = {'C': 7.0, 'H': 11.8333, 'N': 0.5, 'O': 5.16666}
    realigned_spectra = sp.deisotoping_simple(realigned_spectra, tolerance=tolerance_mz, nb_charges=nb_charges, average_distribution={})
    # deisotoped_spectra = sp.deisotoping(np.array(realigned_spectra))
    print(realigned_spectra.shape)
    print("After deisotoping", realigned_spectra[0, 0, ...])
    print("Length =", len(realigned_spectra[0, 0, ...]))

if theoretical:
    species = sr.json_to_species(theoretical)
    ions = [mol for mol in species if mol.category=="Ion"]
    adducts = [mol for mol in species if mol.category=="Adduct"]
    modifications = [mol for mol in species if mol.category=="Modification"]
    theoretical_spectrum = TheoreticalSpectrum(ions, adducts, modifications)
    annotation = si.annotation(realigned_spectra[0, 0, ...].tolist(), theoretical_spectrum.spectrum, tolerance_theoretical)
    peaks = [k for k,v in annotation.items() if len(v) > 0]
    indices_to_width = {k:0 for k in peaks}
    realigned_spectra = np.array(sp.realign_wrt_peaks_mzs(realigned_spectra, np.array(peaks), realigned_spectra[:, 0, ...], indices_to_width))
    print("After alignment with theoretical", len(realigned_spectra[0, 0, ...]), ", length theoretical=", len(theoretical_spectrum.spectrum))

mzs = []
intensities = []
to_array = []
for spectrum in realigned_spectra:
    x, y = spectrum
    mzs.append(x)
    intensities.append(y)

#np.save("data/peaksel_250DJ_prominence75.npy", realigned_spectra)

io.write_imzml(mzs, intensities, p.coordinates, outname)

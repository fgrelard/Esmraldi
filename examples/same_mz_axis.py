"""
Give a common m/z axis
to spectra with different m/z values
"""

import esmraldi.imzmlio as io
import esmraldi.spectraprocessing as sp
import argparse
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input spectrum")
parser.add_argument("-o", "--output", help="Output spectrum")
parser.add_argument("-f", "--factor", help="Multiplicative factor for peak detection", default=150)
parser.add_argument("-l", "--level", help="Noise level", default=2)
parser.add_argument("-n", "--nbpeaks", help="Number of occurrence of peaks across all the spectra for the realignment", default=1)
parser.add_argument("-s", "--step", help="Tolerance step to realign peaks (in m/z)", default=0.02)
args = parser.parse_args()


input_name = args.input
outname = args.output
factor = int(args.factor)
noise_level = float(args.level)
nb_peaks = int(args.nbpeaks)
step_mz = float(args.step)

imzml = io.open_imzml(input_name)
spectra = io.get_spectra(imzml)

peaks = sp.spectra_peak_mzs_adaptative_noiselevel(spectra, factor=factor, noise_level=noise_level, wlen=100)


realigned_spectra = sp.realign_mzs(spectra, peaks, reference="median", nb_occurrence=nb_peaks, step=step_mz)

print(realigned_spectra.shape)

av = sp.spectra_mean(realigned_spectra)

mzs = realigned_spectra[0, 0, :]
plt.plot(mzs, av)
plt.show()

mzs = []
intensities = []
for spectrum in realigned_spectra:
    x, y = spectrum
    mzs.append(x)
    intensities.append(y)

io.write_imzml(mzs, intensities, imzml.coordinates, outname)

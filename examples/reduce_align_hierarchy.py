import argparse
import numpy as np
import esmraldi.imzmlio as io
import matplotlib.pyplot as plt
import esmraldi.spectraprocessing as sp

from esmraldi.peakdetectiontree import PeakDetectionTree

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-s", "--step", help="Step ppm")
parser.add_argument("-o", "--output", help="Output reduced .imzML")
args = parser.parse_args()

input_name = args.input
step = float(args.step)
output_name = args.output

imzml = io.open_imzml(input_name)
spectra = io.get_spectra(imzml)

coordinates = imzml.coordinates
max_x = max(coordinates, key=lambda item:item[0])[0]
max_y = max(coordinates, key=lambda item:item[1])[1]
max_z = max(coordinates, key=lambda item:item[2])[2]
shape = (max_x, max_y, max_z)

mzs = np.unique(np.hstack(spectra[:, 0]))
mean_spectra = sp.spectra_mean_centroided(spectra, mzs)


peak_detection = PeakDetectionTree(mzs, mean_spectra, step)
peaks = peak_detection.extract_peaks()
print(peaks)

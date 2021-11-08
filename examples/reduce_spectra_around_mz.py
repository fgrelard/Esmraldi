import esmraldi.imzmlio as io
import argparse
import numpy as np
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML")
parser.add_argument("-o", "--output", help="Output imzML")
parser.add_argument("-m", "--mz", help="m/z ratio")
parser.add_argument("-t", "--tolerance", help="m/z tolerance", default=1)
args = parser.parse_args()

input_name = args.input
output_name = args.output
value = float(args.mz)
tolerance = float(args.tolerance)

imzml = io.open_imzml(input_name)
spectra = io.get_spectra(imzml)

mz, I = spectra[:, 0], spectra[:, 1]

indices = [np.where((m >= value-tolerance) & (m < value+tolerance))[0] for m in mz]

new_spectra = np.empty(spectra.shape[:2], dtype=object)
for i, ind in enumerate(indices):
    ind_i = indices[i]
    m = mz[i][ind_i]
    intensity = I[i][ind_i]
    if not m.size:
        m, intensity = np.array([0]), np.array([0])
    new_spectra[i, 0] = m
    new_spectra[i, 1] = intensity

new_mzs, new_I = new_spectra[:, 0], new_spectra[:, 1]
coordinates = imzml.coordinates

io.write_imzml(new_mzs, new_I, coordinates, output_name)

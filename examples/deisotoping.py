import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.utils as utils
import esmraldi.imageutils as imageutils
import esmraldi.spectraprocessing as sp
import SimpleITK as sitk
import matplotlib.pyplot as plt
import os
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML or .csv")
parser.add_argument("-d", "--theoretical_diff", help="Theoretical difference", default=1.00335)
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
theoretical_diff = float(args.theoretical_diff)
output_name = args.output

tolerance = 7
if input_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
    spectra = io.get_spectra(imzml)
    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]
    shape = (max_x, max_y, max_z)
    out_spectra = sp.deisotoping_simple_reference(spectra, theoretical_diff, tolerance=tolerance)
    mzs_before = spectra[0, 0, :]
    mzs_after = out_spectra[0, 0, :]
    diff = np.setdiff1d(mzs_before, mzs_after)
    print(diff)
    mzs = out_spectra[:, 0, :]
    intensities = out_spectra[:, 1, :]
    io.write_imzml(mzs, intensities, coordinates, output_name)
    print(out_spectra.shape)
else:
    data = pd.read_excel(input_name)
    values = np.array(data)
    mzs = values[1:, 0]
    means = values[1:, 1::3]
    data_array = np.array([mzs, means[:, 0]])
    print(data_array.shape)
    print(mzs[0])
    indices = sp.deisotoping_reference_indices(data_array, theoretical_diff, tolerance=tolerance)
    mask = np.ones(len(mzs), np.bool)
    mask[indices] = 0
    print(mzs[indices], mzs[mask])
    indices = indices+1
    indices = np.insert(indices, 0, 0)
    values = values[indices, :]
    df = pd.DataFrame(values)
    cols = data.columns
    inds = cols.str.contains("^Unnamed")
    cols = np.array(cols)
    cols[inds] = ""
    df.columns = cols
    df.to_excel(output_name, index=False)

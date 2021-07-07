"""
Convert a 3D imzML file to several 2D imzML
"""

import argparse
import random
import esmraldi.imzmlio as imzmlio
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input 3D imzml")
parser.add_argument("-o", "--output", help="Output directory for 2D imzml")

args = parser.parse_args()

inputname = args.input
outname = args.output

imzml = imzmlio.open_imzml(inputname)
mz, I = imzml.getspectrum(0)
spectra = imzmlio.get_full_spectra(imzml)
max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
max_z = max(imzml.coordinates, key=lambda item:item[2])[2]

image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))
coordinates = imzml.coordinates

print("Parsing all spectra")
intensities, new_coords = imzmlio.get_spectra_from_images(image)
previous_index = 0

array_coords = np.array(new_coords)
array_intensities = np.array(intensities)
split_intensities, split_coords = [], []

print("Splitting")
for i in range(max_z):
    indices = np.where(array_coords.T[2] == i+1)
    current_intensities = list(map(list, array_intensities[indices]))
    current_coordinates = array_coords[indices]

    current_coordinates[...,2] = 1
    current_coordinates = list(map(tuple, current_coordinates))

    mzs = np.tile(mz, (len(indices[0]), 1)).tolist()
    outroot, outext = os.path.splitext(outname)
    current_name = outroot + "_" + str(i) + outext
    print(current_name)
    imzmlio.write_imzml(mzs, current_intensities, current_coordinates, current_name)

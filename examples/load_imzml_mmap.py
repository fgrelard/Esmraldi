import numpy as np

import argparse
import esmraldi.imzmlio as imzmlio
import os
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML")

args = parser.parse_args()
inputname = args.input

print("Open imzML")
imzml = imzmlio.open_imzml(inputname)

print("Building mmap")
imzmlio.build_mmap(imzml)

max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
max_z = max(imzml.coordinates, key=lambda item:item[2])[2]

print("Loading imzML")
mdict = imzmlio.load_mmap(imzml)
full_spectra = mdict["full_spectra"]

im = imzmlio.get_images_from_spectra(full_spectra, (max_x, max_y, max_z))
print(type(im))

plt.imshow(im[..., 0])
plt.show()

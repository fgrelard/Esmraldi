import numpy as np
import sys
import argparse
import esmraldi.imzmlio as imzmlio
import os
import io
import matplotlib.pyplot as plt
from esmraldi.sparsematrix import SparseMatrix
import time

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML")

args = parser.parse_args()
inputname = args.input

print("Open imzML")
imzml = imzmlio.open_imzml(inputname)

print("Building mmap")
imzmlio.build_h5(imzml)

max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
max_z = max(imzml.coordinates, key=lambda item:item[2])[2]

print("Loading imzML")
h5 = imzmlio.load_h5(imzml)

print(type(h5["coordinates"]), h5["shape"][:])
full_spectra = SparseMatrix(h5["coordinates"], h5["data"], h5["shape"].tolist(), sorted=True, has_duplicates=False)
print(mdict.keys())
print(type(full_spectra))

im = mdict["image"]
print(type(im))

plt.imshow(im[..., 3])
plt.show()

import numpy as np
import sys
import argparse
import esmraldi.imzmlio as imzmlio
import os
import io
import matplotlib.pyplot as plt
from esmraldi.sparsematrix import SparseMatrix
from mmappickle.dict import mmapdict
from mmappickle.stubs import EmptyNDArray
from mmappickle.picklers import ArrayPickler
from mmappickle.utils import *
import time

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

print(type(mdict["coordinates"]))
full_spectra = SparseMatrix(mdict["coordinates"], mdict["data"], mdict["shape"], sorted=True, has_duplicates=False)
print(mdict.keys())
print(type(full_spectra))

im = mdict["image"]
print(type(im))

plt.imshow(im[..., 3])
plt.show()

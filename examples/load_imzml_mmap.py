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
import esmraldi.spectraprocessing as sp
import time

def reduce_spectra(full_spectra):
    bytes_dtype = 8
    max_len = 256e9 // (full_spectra.shape[0] * bytes_dtype)
    step, npoints = sp.min_step(full_spectra, max_len)
    print(max_len, step, npoints)
    mdict["reduced_spectra"] = EmptyNDArray(full_spectra.shape[:-1] + (npoints,))
    sp.realign_reducing(mdict["reduced_spectra"], full_spectra, step)

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input imzML")

args = parser.parse_args()
inputname = args.input

print("Open imzML")
imzml = imzmlio.open_imzml(inputname)

max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
max_z = max(imzml.coordinates, key=lambda item:item[2])[2]

print("Building mmap")
imzmlio.build_mmap(imzml)

exit(0)

print("Loading mmap")
mdict = mmapdict(imzmlio.get_filename_mmap(inputname))
print(mdict["shape"], mdict["coordinates"].shape)
full_spectra = SparseMatrix(mdict["coordinates"], mdict["data"], mdict["shape"], sorted=True, has_duplicates=False)

# print(len(peaks))
print(full_spectra[0, 0, :].shape)
if "image" not in mdict:
    imzmlio.get_images_from_spectra_mmap(full_spectra, (max_x, max_y, max_z), mdict)
print(mdict.keys())
print(type(full_spectra))

im = mdict["image"]
# print(type(im))

plt.imshow(im[..., 700].T)
plt.show()

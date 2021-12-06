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


print("Loading imzML")
# mdict = imzmlio.load_mmap(imzml)
mdict = mmapdict(imzmlio.get_filename_mmap(inputname))
mzs = mdict["spectra"][:, 0]


full_spectra = SparseMatrix(mdict["coordinates"], mdict["data"], mdict["shape"], sorted=True, has_duplicates=False)
print(mdict.keys())
print(type(full_spectra))

im = mdict["image"]
print(type(im))

plt.imshow(im[..., 3])
plt.show()

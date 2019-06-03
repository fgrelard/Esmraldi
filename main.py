import pyimzml
import argparse
import src.spectraprocessing as sp
import pyimzml.ImzMLParser as imzmlparser
import matplotlib.pyplot as plt
import src.imzmlio as imzmlio
import numpy as np
import sys

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI (.imzML)")
parser.add_argument("-o", "--output", help="Output")
args = parser.parse_args()

inputname = args.input
outname = args.output

p = imzmlio.open_imzml(inputname)
print(p.imzmldict)
mzs, intensities, coordinates = [], [], []
indices = []

def realignment(current, reference, max_dist=1):
    for elem in reference:
        if abs(current - elem) <= max_dist:
            return elem
    return current

print(len(p.coordinates))
for i, coords in enumerate(p.coordinates):
    x, y = p.getspectrum(i)
    indices_i = sp.peak_indices(y)
    #indices_i = [realignment(index, indices) for index in indices_i]
    indices = indices + indices_i.tolist()


counts = np.bincount(indices)
print(counts.argmax(), " ", counts.max())
print(len(p.coordinates))
plt.plot(range(len(counts)), counts)
plt.show()
indices = np.unique(indices)

for i_image, coords in enumerate(p.coordinates):
    x, y = p.getspectrum(i_image)
    mzs.append(x[indices])
    intensities.append(y[indices])
    coordinates.append(coords)

imzmlio.write_imzml(mzs, intensities, coordinates, outname)


# im = imzmlparser.getionimage(p, 701.316, tol=0.1)
# plt.imshow(im, cmap='jet').set_interpolation('nearest')
# plt.show()

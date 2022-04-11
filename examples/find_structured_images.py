"""
Segmentation example of MALDI images

Finds a relevant subset of image,
i.e. spatially coherent images
and applies a region growing algorithm
to obtain a complete segmentation
on this subset
"""

import esmraldi.segmentation as seg
import esmraldi.imzmlio as imzmlio
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import argparse
import os
import math

from esmraldi.msimage import MSImage

from skimage import segmentation
from skimage import measure
from sklearn.decomposition import PCA

from skimage.filters import gaussian
from skimage.morphology import binary_erosion, opening, disk
from skimage.filters import threshold_otsu, rank
from sklearn import manifold

from scipy import ndimage

def display_stack(img):
    n = img.shape[-1]
    w = math.ceil(math.sqrt(n))
    fig, ax = plt.subplots(w, w)

    for i in range(n):
        ndindex = np.unravel_index(i, ax.shape)
        ax[ndindex].imshow(img[..., i])
    plt.show()



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input (.nii or imzML)")
parser.add_argument("-o", "--output", help="Output segmentation (.nii)")
parser.add_argument("-f", "--factor", help="Factor for the spatially coherent images")
parser.add_argument("-q", "--quantiles", nargs="+", type=int, help="Quantile lower thresholds", default=[60, 70, 80, 90])
parser.add_argument("-u", "--quantile_upper", help="Quantile upper threshold", default=100)
args = parser.parse_args()

inputname = args.input
outname = args.output
factor = float(args.factor)
quantiles = args.quantiles
quantile_upper = int(args.quantile_upper)

radius = 1
selem = disk(radius)


if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    mz, I = imzml.getspectrum(0)
    spectra = imzmlio.get_spectra(imzml)
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
    full_spectra = imzmlio.get_full_spectra(imzml)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]
    shape = (max_x, max_y, max_z)
    img_data = imzmlio.get_images_from_spectra(full_spectra, shape)
else:
    image = nib.load(inputname)
    img_data = image.get_data()

print(img_data.shape)

similar_images, indices = seg.find_similar_images_spatial_coherence(img_data, factor, quantiles=quantiles, upper=quantile_upper, fn=seg.median_perimeter)
# similar_images = seg.find_similar_images_spatial_chaos(img_data, factor, quantiles=[60, 70, 80, 90])
# similar_images = seg.find_similar_images_variance(img_data, factor)
print(similar_images, mzs[indices])

display_stack(similar_images)

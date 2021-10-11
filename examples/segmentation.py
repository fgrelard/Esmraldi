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
parser.add_argument("-t", "--threshold", help="Lower threshold for region growing", default=60)
parser.add_argument("-q", "--quantiles", nargs="+", type=int, help="Quantile lower thresholds", default=[60, 70, 80, 90])
parser.add_argument("-u", "--quantile_upper", help="Quantile upper threshold", default=100)
parser.add_argument("--fill_holes", help="Fill holes in the image.", default=0)
parser.add_argument("--tolerance", help="m/z tolerance value", default=0)
args = parser.parse_args()

threshold = int(args.threshold)
inputname = args.input
outname = args.output
factor = float(args.factor)
quantiles = args.quantiles
quantile_upper = int(args.quantile_upper)
fill_holes = int(args.fill_holes)
tolerance = float(args.tolerance)

radius = 1
selem = disk(radius)


if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    mz, I = imzml.getspectrum(0)
    spectra = imzmlio.get_full_spectra(imzml)
    img_data = MSImage(spectra, coordinates=imzml.coordinates, tolerance=tolerance)
else:
    image = nib.load(inputname)
    img_data = image.get_data()


padding = 3
list_padding = [(padding, padding) for i in range(len(img_data.shape)-1)] + [(0,0)]
img_data = np.pad(img_data, list_padding, 'constant')
print(type(img_data))

similar_images = seg.find_similar_images_spatial_coherence_percentage(img_data, factor, quantiles=quantiles, upper=quantile_upper)
# similar_images = seg.find_similar_images_spatial_chaos(img_data, factor, quantiles=[60, 70, 80, 90])
# similar_images = seg.find_similar_images_variance(img_data, factor)
print(similar_images.shape)
similar_images = seg.sort_size_ascending(similar_images, threshold)
mean_image = np.uint8(cv.normalize(np.average(similar_images, axis=2), None, 0, 255, cv.NORM_MINMAX))
max_image = np.uint8(cv.normalize(np.amax(similar_images, axis=2), None, 0, 255, cv.NORM_MINMAX))

otsu = threshold_otsu(mean_image)
labels = measure.label(mean_image > otsu, background=0)
regionprop = seg.properties_largest_area_cc(labels)
largest_cc = seg.region_property_to_cc(labels, regionprop)
seeds = set(((int(coord[0]), int(coord[1])) for coord in regionprop.coords))

list_end, evolution_segmentation = seg.region_growing(similar_images, seeds, threshold)



x = [elem[0] for elem in list_end]
y = [elem[1] for elem in list_end]
mask = np.ones_like(mean_image)
mask[x, y] = 0
mask = opening(mask, selem)
if fill_holes > 0:
    mask = 1 - ndimage.morphology.binary_fill_holes(1 - mask, structure=np.ones((fill_holes, fill_holes)))

masked_mean_image = np.ma.array(mean_image, mask=mask)
masked_mean_image = masked_mean_image.filled(0)
masked_mean_image = masked_mean_image[padding:-padding, padding:-padding]



fig, ax = plt.subplots(1,3)
ax[0].imshow(mean_image.T)
ax[1].imshow(mask.T)
ax[2].imshow(masked_mean_image.T)
plt.show()

nibimg_similar = nib.Nifti1Image(similar_images, np.eye(4))
similar_name = os.path.splitext(outname)[0] + "_relevantset.nii"
nibimg_similar.to_filename(similar_name)

nibimg = nib.Nifti1Image(masked_mean_image, np.eye(4))
nibimg.to_filename(outname)

nibimg_evolution = nib.Nifti1Image(evolution_segmentation, np.eye(4))
evolution_name = os.path.splitext(outname)[0] + "_evolution.nii"
nibimg_evolution.to_filename(evolution_name)

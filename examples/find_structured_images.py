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
import esmraldi.imageutils as imageutils
import esmraldi.fusion as fusion
import sys
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import argparse
import os
import math

from esmraldi.msimage import MSImage
from esmraldi.sliceviewer import SliceViewer

from skimage import segmentation
from skimage import measure
from sklearn.decomposition import PCA

from skimage.filters import gaussian
from skimage.morphology import binary_erosion, opening, disk
from skimage.filters import threshold_otsu, rank
from sklearn import manifold
from skimage.morphology import convex_hull_image

from scipy import ndimage
from skimage.color import rgb2gray


def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    mask = rgb2gray(mask)
    mask = mask.T
    return mask


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input (.nii or imzML)")
parser.add_argument("--on_sample", help="Whether to consider only on-sample points", action="store_true")
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output segmentation (.nii)")
parser.add_argument("-f", "--factor", help="Factor for the spatially coherent images")
parser.add_argument("-q", "--quantiles", nargs="+", type=int, help="Quantile lower thresholds", default=[60, 70, 80, 90])
parser.add_argument("-u", "--quantile_upper", help="Quantile upper threshold", default=100)
args = parser.parse_args()

inputname = args.input
on_sample = args.on_sample
outname = args.output
factor = float(args.factor)
quantiles = args.quantiles
quantile_upper = int(args.quantile_upper)
mask_name = args.mask
region_names = args.regions
normalization = float(args.normalization)

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
    image = sitk.ReadImage(inputname)
    img_data = sitk.GetArrayFromImage(image).T
    mzs = np.loadtxt(os.path.splitext(inputname)[0] + ".csv")

print(img_data.shape)

if on_sample:
    large_structure_images, ind = seg.find_remote_from_image_edges(img_data, 0.75, quantiles=quantiles, return_indices=True)
    fig, ax = plt.subplots(1)
    tracker = SliceViewer(ax, np.transpose(large_structure_images, (2, 1, 0)))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()



if mask_name and region_names:
    mask = read_image(mask_name)
    regions = []
    for region_name in region_names:
        region = read_image(region_name)
        regions.append(region)

    norm_img = None
    if normalization > 0:
        norm_img = imageutils.get_norm_image(img_data, normalization, mzs)

    indices, indices_ravel = fusion.roc_indices(mask, mask.shape, norm_img)

    region_bool = fusion.region_to_bool(regions, indices_ravel, mask.shape)
    roc_auc_scores = fusion.roc_auc_analysis(img_data, indices, region_bool, norm_img)
    value = 0.7
    print(roc_auc_scores)
    cond = (roc_auc_scores > 1 - value) & (roc_auc_scores < value)
    indices_roc = np.all(cond, axis=-1)
    indices_roc = np.where(indices_roc)[0]

similar_images, indices = seg.find_similar_images_dispersion(img_data, factor, quantiles=quantiles, in_sample=True, return_indices=True)

indices = np.where(indices)[0]
indices = np.intersect1d(indices, indices_roc)
similar_images = img_data[..., indices]

# similar_images, indices = seg.find_similar_images_variance(img_data, factor, return_indices=True)
# similar_images, indices = seg.find_similar_images_spatial_coherence(img_data, factor, quantiles=quantiles, upper=quantile_upper, fn=seg.median_perimeter)
# similar_images = seg.find_similar_images_spatial_chaos(img_data, factor, quantiles=[60, 70, 80, 90])
# similar_images = seg.find_similar_images_variance(img_data, factor)
print(mzs[indices], mzs[indices].shape)

fig, ax = plt.subplots(1)
tracker = SliceViewer(ax, np.transpose(similar_images, (2, 1, 0)))
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

root, ext = os.path.splitext(outname)

sitk.WriteImage(sitk.GetImageFromArray(similar_images), outname)
imzmlio.to_csv(mzs[indices], root + ".csv")

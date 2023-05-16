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
import pandas as pd

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
from skimage.restoration import estimate_sigma

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
parser.add_argument("-r", "--roc", help="ROC file (.xlsx)")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output segmentation (.nii)")
parser.add_argument("-f", "--factor", help="Factor for the spatially coherent images")
parser.add_argument("-q", "--quantiles", nargs="+", type=int, help="Quantile lower thresholds", default=[60, 70, 80, 90])
parser.add_argument("-u", "--quantile_upper", help="Quantile upper threshold", default=100)
parser.add_argument("--names", help="Names to restrict ROC", nargs="+", default=None)
args = parser.parse_args()

inputname = args.input
on_sample = args.on_sample
outname = args.output
factor = float(args.factor)
quantiles = args.quantiles
quantile_upper = int(args.quantile_upper)
roc_name = args.roc
normalization = float(args.normalization)
roc_names = args.names

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
    mzs = np.loadtxt(os.path.splitext(inputname)[0] + ".csv").astype(float)

print(img_data.shape)

norm_img = None
if normalization > 0:
    norm_img = imageutils.get_norm_image(img_data, normalization, mzs)
    for i in range(img_data.shape[-1]):
        img_data[..., i] = imageutils.normalize_image(img_data[..., i], norm_img)


img_data = imzmlio.normalize(img_data)


roc_values_df = pd.read_excel(roc_name)
roc_auc_scores = np.array(roc_values_df).T
names = roc_auc_scores[0, :]
if roc_names is None:
    end = 4
    end = roc_auc_scores.shape[-1]
    ind_names = np.arange(end).astype(int)
else:
    ind_names = np.array([n in roc_names for n in names])
roc_auc_scores = roc_auc_scores[1:, ind_names]
value = 0.8
cond = (roc_auc_scores > 1 - value) & (roc_auc_scores < value)
indices_roc = np.all(cond, axis=-1)
indices_roc = np.where(indices_roc)[0]
print(indices_roc.size)


similar_images, value_array, indices, off_sample_image, off_sample_cond, thresholds = seg.find_similar_images_dispersion_peaks(img_data, factor, quantiles=quantiles, in_sample=True, return_indices=True, return_thresholds=True)


sigmas = []
for i in range(img_data.shape[-1]):
    stdI = imageutils.stddev_image(img_data[..., i])
    nb_pixels = np.count_nonzero(stdI)
    if nb_pixels == 0:
        nb_pixels = 1
    s = np.sum(stdI)/nb_pixels
    sigmas.append(s)
# np.savetxt("test.csv", value_array, delimiter=",", newline=" ")
plt.imsave("off_sample.png", off_sample_image.T)

fig, ax = plt.subplots(1)
label = np.vstack((mzs, off_sample_cond, value_array, sigmas, indices)).T
tracker = SliceViewer(ax, np.transpose(img_data, (2, 1, 0)), labels=label)
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()


indices = np.where(indices)[0]
indices = np.intersect1d(indices, indices_roc)
similar_images = img_data[..., indices]
thresholds = thresholds[indices]

image_thresholded = similar_images.copy()
image_thresholded[image_thresholded < thresholds] = 0
print(image_thresholded.shape, thresholds.shape)

# similar_images, indices = seg.find_similar_images_variance(img_data, factor, return_indices=True)
# similar_images, indices = seg.find_similar_images_spatial_coherence(img_data, factor, quantiles=quantiles, upper=quantile_upper, fn=seg.median_perimeter)
# similar_images = seg.find_similar_images_spatial_chaos(img_data, factor, quantiles=[60, 70, 80, 90])
# similar_images = seg.find_similar_images_variance(img_data, factor)
print(mzs[indices], mzs[indices].shape, image_thresholded.shape, similar_images.shape)

fig, ax = plt.subplots(1)
tracker = SliceViewer(ax, np.transpose(similar_images, (2, 1, 0)))
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

root, ext = os.path.splitext(outname)

print(image_thresholded.dtype, similar_images.dtype)
imzmlio.to_tif(image_thresholded.T,  mzs[indices], root + "_binary.tif")
imzmlio.to_tif(similar_images.T,  mzs[indices], outname)
imzmlio.to_csv(mzs[indices], root + ".csv")
imzmlio.to_csv(mzs[indices], root + "_binary.csv")

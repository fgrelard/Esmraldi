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


def onpick(event, tracker):
    tracker.ind = event.ind[0]
    tracker.update()


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input (.nii or imzML)")
parser.add_argument("--on_sample", help="Whether to consider only on-sample points", action="store_true")
parser.add_argument("-r", "--roc", help="ROC file (.xlsx)")
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=None)
parser.add_argument("-o", "--output", help="Output segmentation (.nii)")
parser.add_argument("-f", "--factor", help="Factor for the spatially coherent images")
parser.add_argument("--offsample_threshold", help="Threshold for offsample", default=0.1)
parser.add_argument("-q", "--quantiles", nargs="+", type=int, help="Quantile lower thresholds", default=[60, 70, 80, 90])
parser.add_argument("-u", "--quantile_upper", help="Quantile upper threshold", default=100)
parser.add_argument("--names", help="Names to restrict ROC", nargs="+", default=None)
parser.add_argument("--size_se", help="Size SE for off sample image", default=5)
args = parser.parse_args()

inputname = args.input
on_sample = args.on_sample
outname = args.output
factor = float(args.factor)
quantiles = args.quantiles
quantile_upper = int(args.quantile_upper)
roc_name = args.roc
normalization = args.normalization
roc_names = args.names
size_se = float(args.size_se)
offsample_threshold = float(args.offsample_threshold)

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

if normalization != None:
    try:
        normalization = float(normalization)
    except:
        pass
    norm_img = imageutils.get_norm_image(img_data, normalization, mzs)
    for i in range(img_data.shape[-1]):
        img_data[..., i] = imageutils.normalize_image(img_data[...,i], norm_img)

img_data = imzmlio.normalize(img_data)

if roc_name is not None:
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

zero_array = np.zeros_like(mzs)
off_sample_cond, value_array, indices = zero_array, zero_array, zero_array
off_sample_image = None

# similar_images, value_array, indices, off_sample_image, off_sample_cond, thresholds = seg.find_similar_images_dispersion_peaks(img_data, factor, quantiles=quantiles, in_sample=True, return_indices=True, return_thresholds=True,  size_elem=size_se)

# similar_images, value_array, indices, off_sample_image, off_sample_cond = seg.find_similar_images_dispersion(img_data, factor, quantiles=quantiles, in_sample=True, return_indices=True)

similar_images, value_array, indices, off_sample_image, off_sample_cond, thresholds = seg.find_similar_images_distance_map(img_data, mzs, factor, offsample_threshold, quantiles=quantiles, in_sample=True, return_indices=True, return_thresholds=True, normalize_max=True)

# similar_images, value_array, _ = seg.find_similar_images_spatial_chaos(img_data, factor, quantiles=quantiles, return_indices=True)
# indices = (value_array < factor) & (off_sample_cond < 0.1)
# similar_images = img_data[..., indices]

# sigmas = []
# for i in range(img_data.shape[-1]):
#     stdI = imageutils.stddev_image(img_data[..., i])
#     nb_pixels = np.count_nonzero(stdI)
#     if nb_pixels == 0:
#         nb_pixels = 1
#     s = np.sum(stdI)/nb_pixels
#     sigmas.append(s)
# np.savetxt("test.csv", value_array, delimiter=",", newline=" ")

if off_sample_image is not None:
    plt.imsave("off_sample.png", off_sample_image.T)

np.set_printoptions(suppress=True)
fig, ax = plt.subplots(1, 2)
print(off_sample_cond.size, value_array.size)
ax[0].scatter(off_sample_cond, value_array, picker=True)
label = np.vstack((mzs, off_sample_cond, value_array, indices, thresholds)).T
tracker = SliceViewer(ax[1], np.transpose(img_data, (2, 1, 0)), labels=label)
cid = fig.canvas.mpl_connect('pick_event', lambda event:onpick(event, tracker))
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

np.savetxt("values.csv", label, delimiter=",", comments="", header="mzs,off_sample,distance,indices,thresholds")

indices = np.where(indices)[0]



# similar_images, indices = seg.find_similar_images_variance(img_data, factor, return_indices=True)
# similar_images, indices = seg.find_similar_images_spatial_coherence(img_data, factor, quantiles=quantiles, upper=quantile_upper, fn=seg.median_perimeter)
# similar_images = seg.find_similar_images_spatial_chaos(img_data, factor, quantiles=[60, 70, 80, 90])
# similar_images = seg.find_similar_images_variance(img_data, factor)

fig, ax = plt.subplots(1)
tracker = SliceViewer(ax, np.transpose(similar_images, (2, 1, 0)))
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

root, ext = os.path.splitext(outname)
imzmlio.to_tif(similar_images.T,  mzs[indices], outname)
imzmlio.to_csv(mzs[indices], root + ".csv")


if thresholds is not None:
    if roc_name is not None:
        indices = np.intersect1d(indices, indices_roc)
    similar_images = img_data[..., indices]

    thresholds = thresholds[indices]

    image_thresholded = similar_images.copy()
    image_thresholded[image_thresholded < thresholds] = 0
    print(image_thresholded.shape, thresholds.shape)
    imzmlio.to_tif(image_thresholded.T,  mzs[indices], root + "_binary.tif")
    imzmlio.to_csv(mzs[indices], root + "_binary.csv")

"""
Segmentation example of MALDI images

Finds a relevant subset of image,
i.e. spatially coherent images
and applies a region growing algorithm
to obtain a complete segmentation
on this subset
"""

import src.segmentation as seg
import src.imzmlio as imzmlio
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import argparse
import os

from skimage import segmentation
from skimage import measure
from sklearn.decomposition import PCA

from skimage.filters import gaussian
from skimage.morphology import binary_erosion, opening, disk
from skimage.filters import threshold_otsu, rank
from sklearn import manifold


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .nii")
parser.add_argument("-o", "--output", help="Output segmentation")
parser.add_argument("-t", "--threshold", help="Lower threshold for region growing", default=60)
args = parser.parse_args()

threshold = int(args.threshold)
inputname = args.input
outname = args.output

radius = 1
selem = disk(radius)

image = nib.load(inputname)

img_data = image.get_data()
padding = 3
img_data = np.pad(img_data, (padding,padding), 'constant')

factor_variance = 0.05
similar_images = seg.find_similar_images_variance(img_data, factor_variance)
mean_image = np.uint8(cv.normalize(np.average(similar_images, axis=2), None, 0, 255, cv.NORM_MINMAX))
otsu = threshold_otsu(mean_image)
labels = measure.label(mean_image > otsu, background=0)
regionprop = seg.properties_largest_area_cc(labels)
largest_cc = seg.region_property_to_cc(labels, regionprop)
seeds = set(((int(coord[0]), int(coord[1])) for coord in regionprop.coords))

list_end = seg.region_growing(similar_images, seeds, threshold)
x = [elem[0] for elem in list_end]
y = [elem[1] for elem in list_end]
mask = np.ones_like(mean_image)
mask[x, y] = 0
mask = opening(mask, selem)
plt.imshow(mask)
plt.show()
masked_mean_image = np.ma.array(mean_image, mask=mask)
masked_mean_image = masked_mean_image.filled(0)
masked_mean_image = masked_mean_image[padding:-padding, padding:-padding]
fig, ax = plt.subplots(1,2)
ax[0].imshow(mean_image.T)
ax[1].imshow(masked_mean_image.T)
plt.show()

nibimg_similar = nib.Nifti1Image(similar_images, np.eye(4))
similar_name = os.path.splitext(outname)[0] + "_relevantset.nii"
nibimg_similar.to_filename(similar_name)

nibimg = nib.Nifti1Image(masked_mean_image, np.eye(4))
nibimg.to_filename(outname)

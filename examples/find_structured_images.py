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
import sys
import SimpleITK as sitk
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
from skimage.morphology import convex_hull_image

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
    image = sitk.ReadImage(inputname)
    img_data = sitk.GetArrayFromImage(image).T
    mzs = np.loadtxt(os.path.splitext(inputname)[0] + ".csv")

print(img_data.shape)

# index = np.argmin(np.abs(mzs - 377.06045532))
# image2D = img_data[..., index]
# norm_img = np.uint8(cv.normalize(image2D, None, 0, 255, cv.NORM_MINMAX))
# upper_threshold = np.percentile(norm_img, 100)
# for quantile in quantiles:
#     threshold = int(np.percentile(norm_img, quantile))
#     mask = (norm_img > threshold) & (norm_img <= upper_threshold)
#     curimg = norm_img.copy()
#     curimg[~mask] = 0
#     print(np.count_nonzero(curimg)/np.prod(curimg.shape))
#     plt.imshow(curimg)
#     plt.show()


values = []
for i in range(img_data.shape[-1]):
    curimg = img_data[..., i]
    curimg = np.uint8(cv.normalize(curimg, None, 0, 255, cv.NORM_MINMAX))
    varimg = imageutils.variance_image(curimg, size=5)
    varimg = np.uint8(cv.normalize(varimg, None, 0, 255, cv.NORM_MINMAX))
    upper_threshold = np.percentile(varimg, 100)
    centroid = [varimg.shape[0]//2, varimg.shape[1]//2]
    thimg = varimg.copy()
    thimg[:] = 1
    thdiff = np.linalg.norm(np.argwhere(thimg) - centroid, axis=-1)
    thstd = np.std(thdiff)
    minvalue = sys.maxsize
    for quantile in quantiles:
        threshold = int(np.percentile(curimg, quantile))
        mask = (curimg > threshold) & (curimg <= upper_threshold)
        binaryimg = curimg.copy()
        binaryimg[mask] = 1
        binaryimg[~mask] = 0
        moments = measure.moments(binaryimg, order=1)

        centroid = [moments[1, 0]/moments[0,0], moments[0, 1]/moments[0,0]]
        ind = np.argwhere(mask)
        max_distance = np.linalg.norm(centroid)
        diff = np.linalg.norm(ind - centroid, axis=-1)
        variance = np.std(diff) / thstd
        if variance < minvalue:
            minvalue = variance
    values.append(minvalue)
value_array = np.array(values)
indices = (value_array < factor)
similar_images = img_data[..., indices]

# similar_images, indices = seg.find_similar_images_variance(img_data, factor, return_indices=True)
# similar_images, indices = seg.find_similar_images_spatial_coherence(img_data, factor, quantiles=quantiles, upper=quantile_upper, fn=seg.median_perimeter)
# similar_images = seg.find_similar_images_spatial_chaos(img_data, factor, quantiles=[60, 70, 80, 90])
# similar_images = seg.find_similar_images_variance(img_data, factor)
print(mzs[indices], mzs[indices].shape)

from esmraldi.sliceviewer import SliceViewer
fig, ax = plt.subplots(1)
tracker = SliceViewer(ax, np.transpose(similar_images, (2, 1, 0)))
fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
plt.show()

root, ext = os.path.splitext(outname)

sitk.WriteImage(sitk.GetImageFromArray(similar_images), outname)
imzmlio.to_csv(mzs[indices], root + ".csv")

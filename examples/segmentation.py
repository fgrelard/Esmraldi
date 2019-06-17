import src.segmentation as seg
import src.imzmlio as imzmlio
import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

from skimage import segmentation
from skimage import measure
from sklearn.decomposition import PCA

from skimage.filters import gaussian
from skimage.morphology import binary_erosion, closing, disk
from skimage.filters import threshold_otsu, rank
from sklearn import manifold
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-t", "--threshold", help="Lower threshold for region growing")
args = parser.parse_args()

threshold = args.threshold

radius = 1
selem = disk(radius)

image = nib.load("/mnt/d/MALDI/imzML/MSI_20190419_01/00/peaksel.nii")

img_data = image.get_data()

img_data = np.pad(img_data, (3,3), 'constant')
similar_images = seg.find_similar_images(img_data)
mean_image = np.uint8(cv.normalize(np.average(similar_images, axis=2), None, 0, 255, cv.NORM_MINMAX))
otsu = threshold_otsu(mean_image)
labels = measure.label(mean_image > otsu, background=0)
regionprop = seg.properties_largest_area_cc(labels)
largest_cc = seg.region_property_to_cc(labels, regionprop)
seeds = set(((int(coord[0]), int(coord[1])) for coord in regionprop.coords))

list_end = seg.region_growing(similar_images, seeds, threshold)
x = [elem[0] for elem in list_end]
y = [elem[1] for elem in list_end]
new_image = np.zeros_like(mean_image)
new_image[x, y] = 1
new_image = closing(new_image, selem)
fig, ax = plt.subplots(1,2)
ax[0].imshow(mean_image.T)
ax[1].imshow(new_image.T)
plt.show()

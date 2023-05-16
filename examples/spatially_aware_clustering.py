"""
Implementation of "Spatially aware clustering"
By Alexandrov et al. (2011)
"""

import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import esmraldi.segmentation as seg
import esmraldi.imzmlio as imzmlio
import esmraldi.fusion as fusion
import esmraldi.imageutils as imageutils
import scipy.ndimage as ndimage
import SimpleITK as sitk

from esmraldi.fastmap import FastMap
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def mapping_neighbors(image, radius, weights):
    r = radius
    size = 2*r+1
    mapping_matrix = np.zeros(shape=(image.shape[0], image.shape[1], size, size, image.shape[-1]))
    w = weights[..., None]
    for index in np.ndindex(image.shape[:-1]):
        i, j = index
        neighbors = image[i-r:i+r+1, j-r:j+r+1]
        if neighbors.shape[0] != size or neighbors.shape[1] != size:
            continue
        mapping_matrix[index] = neighbors * w
    return mapping_matrix

def gaussian_weights(radius):
    size = 2*radius+1
    sigma = size/4
    return np.array([[np.exp((-i**2-j**2)/(2*sigma**2)) for i in range(-radius,radius+1)] for j in range(-radius,radius+1)])

def spatially_aware_clustering(image, k, n, radius):
    weights = gaussian_weights(radius)
    print("Mapping neighbors")
    mapping_matrix = mapping_neighbors(image, radius, weights)
    old_shape = mapping_matrix.shape
    new_shape = (np.prod(old_shape[:-3]), np.prod(old_shape[-3:]))
    fastmap_matrix = mapping_matrix.reshape(new_shape)

    if n < new_shape[-1]:
        fastmap = FastMap(fastmap_matrix, n)
        print("compute projections")
        proj = fastmap.compute_projections()
        print(proj.shape)
        # pd_X = pairwise_distances(fastmap_matrix)**2
        # pd_proj = pairwise_distances(proj)**2
        # print("Sum abs. diff=", np.sum(np.abs(pd_X - pd_proj)))
    else:
        proj = fastmap_matrix

    return proj


def clustering(proj, k, shape):
    kmeans = KMeans(k, random_state=0).fit(proj)
    labels = kmeans.labels_
    image_labels = labels.reshape(shape)
    return image_labels



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-n", "--number", help="Number of dimensions after dimension reduction (fastmap)", default=50)
parser.add_argument("-k", "--classes", help="Number of clusters for kmeans", default=7)
parser.add_argument("-r", "--radius", help="Radius for spatial features", default=1)
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold (optional)", default=0)
parser.add_argument("--normalization", help="Normalize spectra by their norm", default=None)
parser.add_argument("--mask", help="Mask", default=None)
parser.add_argument("--cosine", help="Whether to normalize spectra in order to approximate cosine distance in KMeans computation", action="store_true")

args = parser.parse_args()

input_name = args.input
outname = args.output
radius = int(args.radius)
n = int(args.number)
k = int(args.classes)
threshold = int(args.threshold)
normalization = args.normalization
is_cosine = args.cosine
mask_name = args.mask


if input_name.lower().endswith(".imzml"):
    imzml = io.open_imzml(input_name)
    spectra = io.get_spectra(imzml)
    print(spectra.shape)
    coordinates = imzml.coordinates
    max_x = max(coordinates, key=lambda item:item[0])[0]
    max_y = max(coordinates, key=lambda item:item[1])[1]
    max_z = max(coordinates, key=lambda item:item[2])[2]
    shape = (max_x, max_y, max_z)

    full_spectra = io.get_full_spectra(imzml)
    mzs = np.unique(np.hstack(spectra[:, 0]))
    mzs = mzs[mzs>0]
    print(len(mzs))
    images = io.get_images_from_spectra(full_spectra, shape)
else:
    image_itk = sitk.ReadImage(input_name)
    images = sitk.GetArrayFromImage(image_itk).T
    mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv", encoding="utf-8-sig")

images = images[..., mzs >= threshold]

print("normalization", normalization)
if normalization != None:
    try:
        normalization = float(normalization)
    except:
        pass
    norm_img = imageutils.get_norm_image(images, normalization, mzs)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[...,i], norm_img)

if is_cosine:
    norm_img = imageutils.get_norm_image(images, "norm", None)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[..., i], norm_img)

if mask_name != None:
    mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_name)).T
    images[mask==0] = 0
    plt.imshow(images[..., 1])
    plt.show()




mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

nb_peaks = images.shape[-1]
print("Number of peaks=", nb_peaks)

shape = images.shape

if len(shape) == 4:
    for i in range(shape[-2]):
        current_image = images[..., i, :]
        proj = spatially_aware_clustering(current_image, k, n, radius)
else:
    proj = spatially_aware_clustering(images, k, n, radius)

for nb_cluster in range(2, k+1):
    image_labels = clustering(proj, nb_cluster, images.shape[:-1])
    image_labels_itk = sitk.GetImageFromArray(image_labels.astype(np.uint8).T)
    root, ext = os.path.splitext(outname)
    curroutname = root + "_" + str(nb_cluster) + ext
    print(nb_cluster, curroutname)
    sitk.WriteImage(image_labels_itk, curroutname)

# images = imzmlio.normalize(images)
# outname_csv = os.path.splitext(outname)[0] + ".csv"
# out_array = np.zeros(shape=(nb_peaks, k))
# for i in range(k):
#     indices = np.where(image_labels == i)
#     not_indices = np.where(image_labels != i)
#     median_spectrum = np.median(images[indices], axis=0)
#     print(median_spectrum.shape)
#     other_median_spectrum = np.median(images[not_indices], axis=0)
#     median_spectrum -= other_median_spectrum
#     top_indices = np.argsort(median_spectrum)[::-1]
#     top_molecules = mzs[top_indices]
#     out_array[:, i] = top_molecules
# np.savetxt(outname_csv, out_array, delimiter=";", fmt="%s")



plt.imshow(image_labels.T)
plt.show()

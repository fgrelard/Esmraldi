import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
import esmraldi.segmentation as seg
import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import esmraldi.imageutils as imageutils

import SimpleITK as sitk
from sklearn.cluster import KMeans

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-k", "--classes", help="Number of clusters for kmeans", default=7)
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold (optional)", default=0)
parser.add_argument("-n", "--normalization", help="Normalize spectra by their norm", default=None)
parser.add_argument("--cosine", help="Whether to normalize spectra in order to approximate cosine distance in KMeans computation", action="store_true")

args = parser.parse_args()

input_name = args.input
out_name = args.output
k = int(args.classes)
threshold = int(args.threshold)
normalization = args.normalization
is_cosine = args.cosine

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
    print("Cosine approximation")
    norm_img = imageutils.get_norm_image(images, "norm", None)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[..., i], norm_img)

image_flatten = fusion.flatten(images, is_spectral=True).T
print(image_flatten.shape)
kmeans = KMeans(k, random_state=0).fit(image_flatten)
labels = kmeans.labels_
image_labels = labels.reshape(images.shape[:-1])

plt.imshow(image_labels.T)
plt.show()

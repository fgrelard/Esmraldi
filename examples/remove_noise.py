"""
Remove noise on MALDI images
(deprecated)
"""
import argparse
import numpy as np

import esmraldi.imzmlio as imzmlio
import SimpleITK as sitk
import matplotlib.pyplot as plt
from esmraldi.sliceviewer import SliceViewer
import esmraldi.fusion as fusion

from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-o", "--output", help="Output image (imzML)")
parser.add_argument("-k", "--number_classes", help="Number of classes for k-means algorithm", default=0)
parser.add_argument("-n", "--number_components", help="Number of components for NMF")
args = parser.parse_args()

inputname = args.input
outname = args.output
k = int(args.number_classes)
n = int(args.number_components)

if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    spectra = imzmlio.get_full_spectra(imzml)
    max_x = max(imzml.coordinates, key=lambda item:item[0])[0]
    max_y = max(imzml.coordinates, key=lambda item:item[1])[1]
    max_z = max(imzml.coordinates, key=lambda item:item[2])[2]
    image = imzmlio.get_images_from_spectra(spectra, (max_x, max_y, max_z))
    mzs, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(inputname)).T
    mzs = [i for i in range(image.shape[2])]
    mzs = np.asarray(mzs)

# shape = image.shape
# new_shape = (np.prod(shape[:-1]), shape[-1])

# image_flat = np.reshape(image, new_shape)
ind =  (np.abs(mzs - 701.2809)).argmin()
print(ind)
for i in range(n):
    current_image = image[..., i, :]
    current_image_pos = np.where(current_image > 0, 1, 0)
    shape = current_image.shape
    new_shape = (np.prod(shape[:-1]), shape[-1])
    image_flat = np.reshape(current_image, new_shape)
    # kmeans = KMeans(k, random_state=0).fit(image_flat)
    # labels = kmeans.labels_
    cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward').fit(image_flat)
    labels = cluster.labels_
    image_labels = labels.reshape(shape[:-1])
    fig, ax = plt.subplots(1,2)
    ax[0].imshow(image_labels)
    ax[1].imshow(current_image[..., ind])
    plt.show()

print("Clustering", new_shape)
kmeans = KMeans(k, random_state=0).fit(image_nmf)
labels = kmeans.labels_
# cluster = AgglomerativeClustering(n_clusters=k, affinity='euclidean', linkage='ward').fit(image_flat)
# labels = cluster.labels_
print(labels.shape)
image_labels = labels.reshape(shape[:-1])
print(image_labels.shape)

if len(shape) == 3:
    plt.imshow(image_labels)
    plt.show()
elif len(shape) == 4:
    fig, ax = plt.subplots(1, 1)
    tracker = SliceViewer(ax,
                          np.transpose(image_labels, (2, 1, 0)))
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

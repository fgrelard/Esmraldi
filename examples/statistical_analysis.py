import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AffinityPropagation
import src.segmentation as seg
import src.imzmlio as imzmlio
import argparse
import nibabel as nib
import SimpleITK as sitk
import math
import os

def plot_clustering(X, labels):
    n_clusters = len(np.unique(labels))
    cm = plt.get_cmap('gist_rainbow')
    for k in range(n_clusters):
        class_members = labels == k
        plt.plot(X[class_members, 0], X[class_members, 1], '.', c=cm(k//3*3.0/33))
    plt.legend(np.unique(labels))
    plt.show()

def clustering(image, fit_pca):
    X_r = fit_pca.transform(image)
    af = AffinityPropagation(preference=-50).fit(X_r)
    return X_r, af

def pca(image):
    pca = PCA(n_components=5)
    fit_pca = pca.fit(image_norm)
    return fit_pca

def weighted_distance(X, weights):
    return np.sqrt(np.sum(X**2 * weights))

def select_images(images, mzs, clustering, weights, top=1):
    labels = clustering.labels_
    centers = clustering.cluster_centers_
    point = fit_pca.transform(mri_norm)

    distances = np.array([weighted_distance(center-point, weights) for center in centers])
    indices = [i for i in range(len(distances))]
    indices.sort(key=lambda x: distances[x])
    indices = np.array(indices)
    condition = np.any(np.array([labels == indices[i] for i in range(top)]), axis=0)
    similar_images = image[..., condition].T
    similar_mzs = mzs[condition]
    return np.float32(similar_images), similar_mzs



parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image")
parser.add_argument("-m", "--mri", help="Input MRI image")
parser.add_argument("-o", "--output", help="Output image")
parser.add_argument("-t", "--top", help="#Top", default=1)
args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
top = int(args.top)

if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    image = imzmlio.get_all_array_images(imzml)
    mzs, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(inputname, sitk.sitkFloat32)).T
    mzs = [i for i in range(len(image.shape[2]))]

image_mri = sitk.GetArrayFromImage(sitk.ReadImage(mriname, sitk.sitkFloat32)).T

image_norm = seg.preprocess_pca(image)
mri_norm = seg.preprocess_pca(image_mri)
print(image_norm.shape)
fit_pca = pca(image_norm)
weights = fit_pca.explained_variance_ratio_ / np.sum(fit_pca.explained_variance_ratio_)
weights = [1 for i in range(len(weights))]
X_r, af = clustering(image_norm, fit_pca)

similar_images, similar_mzs = select_images(image, mzs, af, weights, top)
itk_similar_images = sitk.GetImageFromArray(similar_images)
sitk.WriteImage(itk_similar_images, outname)

outname_csv = os.path.splitext(outname)[0] + ".csv"
np.savetxt(outname_csv, similar_mzs, delimiter=";")

point = fit_pca.transform(mri_norm)
labels = af.labels_
plt.plot(point[0, 0], point[0, 1], "rx")
plot_clustering(X_r, labels)

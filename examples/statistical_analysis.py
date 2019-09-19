import numpy as np
import matplotlib.pyplot as plt
import src.segmentation as seg
import src.imzmlio as imzmlio
import src.fusion as fusion
import argparse
import nibabel as nib
import SimpleITK as sitk
import math
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def plot_clustering(X, labels, mri):
    n_clusters = len(np.unique(labels))
    cm = plt.get_cmap('nipy_spectral')
    plt.plot(mri[0][0], mri[0][1], "x")
    for k in range(n_clusters):
        class_members = labels == k
        plt.plot(X[class_members, 0], X[class_members, 1], '.', c=cm(float(k)/n_clusters))
    plt.legend(np.unique(labels), bbox_to_anchor=(1.05,1))
    plt.xlabel("First component")
    plt.ylabel("Second component")
    plt.show()



def plot_pca(X_r, af):
    plt.scatter(X_r[:, 0], X_r[:, 1], c=af.predict(X_r), cmap="nipy_spectral", alpha=0.7, norm=colors.Normalize(0, 25))
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image")
parser.add_argument("-m", "--mri", help="Input MRI image")
parser.add_argument("-o", "--output", help="Output image")
parser.add_argument("-r", "--ratio", help="Compute ratio images", action="store_true")
parser.add_argument("-t", "--top", help="#Top", default=0)
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold", default=0)
args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
is_ratio = args.ratio
top = int(args.top)
threshold = int(args.threshold)

if top <= 0:
    top = None


if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    image = imzmlio.to_image_array(imzml)
    mzs, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(inputname, sitk.sitkUInt8)).T
    mzs = [i for i in range(image.shape[2])]
    mzs = np.asarray(mzs)

image = image[..., mzs >= threshold]
mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

image_mri = sitk.GetArrayFromImage(sitk.ReadImage(mriname, sitk.sitkUInt8)).T

if is_ratio:
    ratio_images, ratio_mzs = fusion.extract_ratio_images(image, mzs)
    image = np.concatenate((image, ratio_images), axis=2)
    mzs = np.concatenate((mzs, ratio_mzs))

image = imzmlio.normalize(image)


image_norm = seg.preprocess_pca(image)
mri_norm = seg.preprocess_pca(image_mri)
print(image_norm.shape)

print("Computing PCA")
fit_pca = fusion.pca(image_norm)

point = fit_pca.transform(mri_norm)
print("Explained variance ratio =", fit_pca.explained_variance_ratio_)

weights = fit_pca.explained_variance_ratio_ / np.sum(fit_pca.explained_variance_ratio_)
X_r = fit_pca.transform(image_norm)
X_train, X_test = fusion.post_processing(X_r, point)
clustering = fusion.clustering_kmeans(X_train)
plot_pca(X_train, clustering)

labels = None

centers = X_train
point_mri = X_test
weights = [1 for i in range(X_test.shape[1])]

if top is not None:
    af = fusion.clustering(image_norm, X_r)
    labels = af.labels_
    centers = af.cluster_centers_

similar_images, similar_mzs, distances = fusion.select_images(image,point_mri, centers, weights,  mzs, labels, None)
print("Selecting images end")

similar_images = similar_images[:1000]
itk_similar_images = sitk.GetImageFromArray(similar_images)
sitk.WriteImage(itk_similar_images, outname)

outname_csv = os.path.splitext(outname)[0] + ".csv"
np.savetxt(outname_csv, np.transpose((similar_mzs, distances)), delimiter=";", fmt="%s")
#np.save("data/labels_maldi.npy", labels)
#plt.plot(point[0, 0], point[0, 1], "rx")

#plot_clustering(X_r, labels, point)

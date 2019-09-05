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
parser.add_argument("-t", "--top", help="#Top", default=1)
args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
is_ratio = args.ratio
top = int(args.top)

if inputname.lower().endswith(".imzml"):
    imzml = imzmlio.open_imzml(inputname)
    image = imzmlio.get_all_array_images(imzml)
    mzs, intensities = imzml.getspectrum(0)
else:
    image = sitk.GetArrayFromImage(sitk.ReadImage(inputname, sitk.sitkUInt8)).T
    mzs = [i for i in range(image.shape[2])]
    mzs = np.asarray(mzs)

image_mri = sitk.GetArrayFromImage(sitk.ReadImage(mriname, sitk.sitkUInt8)).T

image = imzmlio.normalize(image)
if is_ratio:
    ratio_images = fusion.extract_ratio_images(image)
    image = np.concatenate(image, ratio_images)
    z = image.shape[-1]
    mzs += [str(i) + "/" + str(j) for i in range(z-1, 0, -1) for j in range(i)]
    print(mzs[-1])
image_norm = seg.preprocess_pca(image)
mri_norm = seg.preprocess_pca(image_mri)
print(image_norm.shape)

fit_pca = fusion.pca(image_norm)
point = fit_pca.transform(mri_norm)
print("Explained variance ratio =", fit_pca.explained_variance_ratio_)

weights = fit_pca.explained_variance_ratio_ / np.sum(fit_pca.explained_variance_ratio_)
weights = [1 for i in range(len(weights))]

X_r, af = fusion.clustering(image_norm, fit_pca)
labels = af.labels_
centers = af.cluster_centers_
centers = X_r
similar_images, similar_mzs = fusion.select_images(fit_pca, image, mzs, mri_norm, labels, centers, weights, None)
itk_similar_images = sitk.GetImageFromArray(similar_images)
sitk.WriteImage(itk_similar_images, outname)

outname_csv = os.path.splitext(outname)[0] + ".csv"
np.savetxt(outname_csv, similar_mzs, delimiter=",", fmt="%10.5f")
#np.save("data/labels_maldi.npy", labels)
#plt.plot(point[0, 0], point[0, 1], "rx")

plot_clustering(X_r, labels, point)

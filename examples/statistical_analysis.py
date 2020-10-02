"""
Statistical analysis

   1/ Matrix factorization of MALDI datacube
   2/ Projection of the MRI in the space defined
      by the axes of the dimension reduction method
   3/ Sort the MALDI images by ascending order of
      the Euclidean distance between the MALDI image
      and the MRI image in this reduced space
"""
import numpy as np
import matplotlib.pyplot as plt
import esmraldi.segmentation as seg
import esmraldi.imzmlio as imzmlio
import esmraldi.fusion as fusion
import argparse
import nibabel as nib
import SimpleITK as sitk
import math
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF, PCA

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.filters import threshold_otsu
from esmraldi.sliceviewer import SliceViewer


def plot_clustering(X, labels, mri):
    n_clusters = len(np.unique(labels))
    cm = plt.get_cmap('gray')
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

def visualize_scatter_with_images(X_all, images_maldi, images_mri,figsize=(45,45), image_zoom=1):
    fig, ax = plt.subplots(figsize=figsize)
    artists = []
    for xy, i in zip(X_all[:-1, :], images_maldi):
        x0, y0 = xy
        img = OffsetImage(i, zoom=image_zoom, cmap='gray')
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        artists.append(ax.add_artist(ab))

    img_mri = OffsetImage(images_mri, zoom=image_zoom, cmap='gray')
    x_mri, y_mri = X_all[-1:, 0], X_all[-1:, 1]
    ab_mri = AnnotationBbox(img_mri, (x_mri, y_mri), xycoords='data', frameon=True, bboxprops =dict(ec="r", lw=2))
    artists.append(ax.add_artist(ab_mri))
    ax.update_datalim(X_all)
    ax.autoscale()
    plt.show()

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-m", "--mri", help="Input MRI image (ITK format)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-n", "--number", help="Number of components for dimension reduction", default=5)
parser.add_argument("-r", "--ratio", help="Compute ratio images (optional)", action="store_true")
parser.add_argument("-t", "--top", help="#Top (optional)", default=0)
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold (optional)", default=0)
parser.add_argument("--norm", help="Normalization image filename (optional)")
parser.add_argument("--post_process", help="Post process with tSNE (optional)", action="store_true")

args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
is_ratio = args.ratio
n = int(args.number)
top = int(args.top)
threshold = int(args.threshold)
normname = args.norm
post_process = args.post_process

if top <= 0:
    top = None


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

print("Mass-to-charge ratio=", mzs)

image = image[..., mzs >= threshold]

if normname is not None:
    print("Norm image detected")
    norm_img = sitk.ReadImage(normname)
    norm_img = sitk.GetArrayFromImage(norm_img).T
    norm_img_3D = norm_img[..., None]
    before = image.max()
    image = np.divide(image, norm_img_3D, out=np.zeros_like(image, dtype=np.float), where=norm_img_3D!=0)
    after = image.max()
    print("Before max=", before, ", after=", after)


mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)


image_mri = sitk.GetArrayFromImage(sitk.ReadImage(mriname, sitk.sitkFloat32)).T

if is_ratio:
    ratio_images, ratio_mzs = fusion.extract_ratio_images(image, mzs)
    image = np.concatenate((image, ratio_images), axis=2)
    mzs = np.concatenate((mzs, ratio_mzs))

image = imzmlio.normalize(image)
image_norm = fusion.flatten(image, is_spectral=True)

mri_norm = imzmlio.normalize(image_mri)
mri_norm = fusion.flatten(mri_norm)

print(mri_norm.shape)
print(image_norm.shape)

print("Computing Dimension reduction")

# fit_red = fusion.nmf(image_norm, n)
# print(mri_norm.T.shape)
nmf = NMF(n_components=n, init='nndsvda', solver='mu', random_state=0)
# fit_red = nmf.fit(image_norm.T)
# eigenvectors = fit_red.components_ #H
# image_eigenvectors = nmf.transform(image_norm.T); #W
# mri_eigenvectors = nmf.transform(mri_norm.T)

# shape_mri = image_mri.shape + (mri_eigenvectors.shape[-1],)
# mri_eigenvectors = mri_eigenvectors.reshape(shape_mri)



fit_red = nmf.fit(image_norm)
point = fit_red.transform(mri_norm)
X_r = fit_red.transform(image_norm)
image_eigenvectors = nmf.inverse_transform(X_r)
centers = X_r
point_mri = point

image_eigenvectors = image_eigenvectors.T
new_shape = image.shape[:-1] + (image_eigenvectors.shape[-1],)
image_eigenvectors = image_eigenvectors.reshape(new_shape)

print(image_eigenvectors.shape)

print("Explained variance ratio=", fusion.get_score(fit_red, image_norm))

if post_process:
    X_train, X_test = fusion.post_processing(X_r, point)
    centers = X_train
    point_mri = X_test
    clustering = fusion.clustering_kmeans(X_train)
    plt.plot(X_train[:, 0], X_train[:, 1], "b.")
    plt.plot(X_test[:, 0], X_test[:, 1], "ro")
    plt.show()
    plt.close()

if not is_ratio:
    X_all = np.concatenate((centers, point_mri), axis=0)
    tsne_all = StandardScaler().fit_transform(X_all)
    X_r_all = np.concatenate((X_r, point), axis=0)
    pca_all = StandardScaler().fit_transform(X_r_all)
    mri = StandardScaler().fit_transform(point)
    pca_all = pca_all[..., :2]
    size = (100, 100)
    if len(image.shape) == 3:
        images_maldi = [cv2.resize(i, size) for i in image.T]
        image_mri = cv2.resize(image_mri.T, size)
        visualize_scatter_with_images(pca_all,
                                      images_maldi,
                                      image_mri,
                                      figsize=size,
                                      image_zoom=0.7)
    elif len(image.shape) == 4:
        images_maldi = [cv2.resize(i[..., i.shape[-1]//2], size) for i in np.transpose(image, (3, 0, 1, 2))]
        thumbnail_mri = image_mri.copy()
        thumbnail_mri = cv2.resize(thumbnail_mri[..., thumbnail_mri.shape[-1]//2], size)
        visualize_scatter_with_images(pca_all,
                                      images_maldi,
                                      thumbnail_mri,
                                      figsize=size,
                                      image_zoom=0.7)

plt.plot(X_r[:, 0], X_r[:, 1], "b.")
plt.plot(point[:, 0], point[:, 1], "ro")
plt.show()
plt.close()

labels = None
weights = [1 for i in range(centers.shape[1])]

if top is not None:
    af = fusion.clustering_kmeans(image_norm, X_r)
    labels = af.labels_
    centers = af.cluster_centers_

similar_images, similar_mzs, distances = fusion.select_images(image,point_mri, centers, weights,  mzs, labels, None)

print("Selecting images end")

similar_images = similar_images[..., 0:100]

index = np.where(mzs == similar_mzs[0])[0]
w = X_r[index, ...] / np.sum(X_r[index, ...])
image_closest = fusion.get_reconstructed_image_from_components(image_eigenvectors, w)
image_closest = np.transpose(image_closest, (2, 1, 0))
image_closest = imzmlio.normalize(image_closest)

w_mri = point / np.sum(point)
print(w, w_mri)
print(X_r.shape)
print(point, X_r[index, ...])
mri_reconstructed = fusion.get_reconstructed_image_from_components(image_eigenvectors, w_mri.T)
mri_reconstructed = np.transpose(mri_reconstructed, (2, 1, 0))
mri_reconstructed = imzmlio.normalize(mri_reconstructed)

if len(similar_images.shape) == 3:
    fig, ax = plt.subplots(1, 4)
    ax[0].imshow(similar_images[..., 0])
    ax[1].imshow(mri_norm)
    ax[2].imshow(image_closest)
    ax[3].imshow(mri_reconstructed)
    plt.show()
elif len(similar_images.shape) == 4:
    fig, ax = plt.subplots(1, 4)
    tracker = SliceViewer(ax,
                          np.transpose(similar_images[..., 0], (2, 1, 0)),
                          np.transpose(np.reshape(mri_norm, image_mri.shape), (2, 1, 0)),
                          np.transpose(image_closest, (2, 1, 0)),
                          np.transpose(mri_reconstructed, (2, 1, 0)),
                          vmin=0, vmax=255)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

if len(similar_images.shape) == 4:
    s = similar_images.shape
    similar_images = similar_images.reshape(s[0], s[1], s[2]*s[3], order="F")

itk_similar_images = sitk.GetImageFromArray(similar_images.T)
sitk.WriteImage(itk_similar_images, outname)

outname_csv = os.path.splitext(outname)[0] + ".csv"
np.savetxt(outname_csv, np.transpose((similar_mzs, distances)), delimiter=";", fmt="%s")

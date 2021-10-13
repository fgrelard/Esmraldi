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
import esmraldi.registration as reg
import esmraldi.imageutils as utils
import argparse
import nibabel as nib
import SimpleITK as sitk
import math
import os
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import tifffile

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import NMF, PCA

from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.filters import threshold_otsu
from esmraldi.sliceviewer import SliceViewer

from skimage.filters import sobel
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter

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



def statistical_analysis(outname, image, image_mri, mzs, n, is_ratio, top, post_process, is_norm, is_denoise, is_memmap, translate_components):

    if is_denoise:
        size = (5, 5, 1)
        if len(image.shape) == 4:
            size += (1,)
        image = median_filter(image, size=size)

    # image[image_mri == 0] = 0

    if is_ratio:
        ratio_images, ratio_mzs = fusion.extract_ratio_images(image, mzs)
        image = np.concatenate((image, ratio_images), axis=-1)
        mzs = np.concatenate((mzs, ratio_mzs))
        indices_delete = fusion.remove_indices(image)
        image = np.delete(image, indices_delete, axis=-1)
        mzs = np.delete(mzs, indices_delete)

    memmap_dir = os.path.dirname(outname) + os.path.sep + "mmap" + os.path.sep
    memmap_basename = os.path.splitext(os.path.basename(outname))[0]
    memmap_image_filename = memmap_dir + memmap_basename + ".npy"
    memmap_files_exist = (os.path.exists(memmap_dir)
                          and os.path.exists(memmap_image_filename))
    if is_memmap and not memmap_files_exist:
        print("Writing mem-map")
        os.makedirs(memmap_dir, exist_ok=True)
        np.save(memmap_image_filename, image)
    if is_memmap:
        print("Reading mem-map")
        image = np.load(memmap_image_filename, mmap_mode="r")

    image_norm = imzmlio.normalize(image)
    mri_norm = imzmlio.normalize(image_mri)

    print(image_norm.shape)

    if is_norm:
        image_norm = image_norm.astype(np.float64)
        mri_norm = mri_norm.astype(np.float64)

        for index in range(image_norm.shape[-1]):
            current_image  = image_norm[..., index]
            norm =  np.linalg.norm(current_image)
            if norm > 0:
                current_image /= norm
                image_norm[..., index] = current_image
        mri_norm /= np.linalg.norm(mri_norm)
    image_norm = fusion.flatten(image_norm, is_spectral=True)
    mri_norm = fusion.flatten(mri_norm)

    print("Computing Dimension reduction")

    nmf = NMF(n_components=n, init='nndsvda', solver='mu', random_state=0, max_iter=1000)

    # fit_red = nmf.fit(image_norm.T)
    # eigenvectors = fit_red.components_ #H
    # image_eigenvectors = nmf.transform(image_norm.T); #W
    # mri_eigenvectors = nmf.transform(mri_norm.T)

    # shape_mri = image_mri.shape + (mri_eigenvectors.shape[-1],)
    # mri_eigenvectors = mri_eigenvectors.reshape(shape_mri)


    fit_red = nmf.fit(image_norm)

    W = fit_red.transform(image_norm)
    H = fit_red.components_

    image_eigenvectors = H.T
    new_shape = image.shape[:-1] + (image_eigenvectors.shape[-1],)
    image_eigenvectors = image_eigenvectors.reshape(new_shape)

    image_eigenvectors_translated = image_eigenvectors

    H_translated = H
    if translate_components:
        image_eigenvectors_translated, translated_image = reg.register_component_images(image_mri, image, image_eigenvectors, 3)

        H_translated = image_eigenvectors_translated.reshape(H.T.shape).T

    # We use translated components ONLY for MRI reconstruction
    fit_red.components_ = H_translated
    point = fit_red.transform(mri_norm)

    #Normal components for MS images
    fit_red.components_ = H
    X_r = fit_red.transform(image_norm)

    centers = X_r
    point_mri = point


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
            # visualize_scatter_with_images(pca_all,
            #                              images_maldi,
            #                              image_mri,
            #                              figsize=size,
            #                              image_zoom=0.7)
        elif len(image.shape) == 4:
            images_maldi = [cv2.resize(i[..., i.shape[-1]//2], size) for i in np.transpose(image, (3, 0, 1, 2))]
            thumbnail_mri = image_mri.copy()
            thumbnail_mri = cv2.resize(thumbnail_mri[..., thumbnail_mri.shape[-1]//2], size)
            #visualize_scatter_with_images(pca_all,
            #                              images_maldi,
            #                              thumbnail_mri,
            #                              figsize=size,
            #                              image_zoom=0.7)

    # plt.plot(X_r[:, 0], X_r[:, 1], "b.")
    # plt.plot(point[:, 0], point[:, 1], "ro")
    # plt.show()
    # plt.close()

    labels = None
    weights = [1 for i in range(centers.shape[1])]

    if top is not None:
        af = fusion.clustering_kmeans(image_norm, X_r)
        labels = af.labels_
        centers = af.cluster_centers_

    similar_images, similar_mzs, distances = fusion.select_images(image,point_mri, centers, weights,  mzs, labels, None)

    print("Selecting images end")
    diff = fusion.closest_reconstruction(image, X_r, point, image_eigenvectors, image_eigenvectors_translated)
    print(mzs.shape)
    print(similar_mzs[:10], mzs[diff.argsort()][:10])

    similar_images = similar_images[..., 0:100]
    similar_images = imzmlio.normalize(similar_images)

    index_closest = np.where(mzs == similar_mzs[0])
    index = index_closest[0]


    w = X_r[index, ...] / np.sum(X_r[index, ...])

    image_closest = fusion.reconstruct_image_from_components(image_eigenvectors, w.T)
    image_closest = image_closest.T
    image_closest = imzmlio.normalize(image_closest)

    w_mri = point / np.sum(point)
    mri_reconstructed = fusion.reconstruct_image_from_components(image_eigenvectors_translated, w_mri.T)
    mri_reconstructed = mri_reconstructed.T
    i = np.where((mri_reconstructed>0))
    img_tmp = np.reshape(mri_norm, image_mri.shape)
    mri_reconstructed = imzmlio.normalize(mri_reconstructed)
    mri_reconstructed_diff = mri_reconstructed.copy().astype(float)
    if is_norm:
        mri_reconstructed_diff /= np.linalg.norm(mri_reconstructed_diff)
    diff_reconstruction = np.mean(np.abs(mri_reconstructed_diff[i] - img_tmp[i]))/np.max(img_tmp)
    print("Average diff NMF reconstruction (percentage)=", "{:.5f}".format(diff_reconstruction))


    if len(similar_images.shape) == 3:
        fig, ax = plt.subplots(1, 4)
        ax[0].imshow(similar_images[..., 0], cmap="gray")
        ax[1].imshow(np.reshape(mri_norm, image_mri.shape), cmap="gray")
        ax[2].imshow(image_closest, cmap="gray")
        ax[3].imshow(mri_reconstructed, cmap="gray")
        # plt.show()
    elif len(similar_images.shape) == 4:
        fig, ax = plt.subplots(1, 4)

        tracker = SliceViewer(ax,
                              np.transpose(similar_images[..., 0], (2, 1, 0)),
                              np.transpose(np.reshape(mri_norm, image_mri.shape), (2, 1, 0)),
                              np.transpose(image_closest, (2, 1, 0)),
                              np.transpose(mri_reconstructed, (2, 1, 0)),
                              vmin=0, vmax=255)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        # plt.show()

    if len(similar_images.shape) == 4:
        s = similar_images.shape
        similar_images = similar_images.reshape(s[0], s[1], s[2], s[3], order="F")
        tifffile.imwrite(outname, similar_images.T, imagej=True)
    else:
        itk_similar_images = sitk.GetImageFromArray(similar_images.T)
        sitk.WriteImage(itk_similar_images, outname)
    sitk.WriteImage(sitk.GetImageFromArray(mri_reconstructed.T), os.path.splitext(outname)[0] + "_reconstruction.tif")

    outname_csv = os.path.splitext(outname)[0] + ".csv"
    np.savetxt(outname_csv, np.transpose((similar_mzs, distances)), delimiter=";", fmt="%s")


def normalize(image):
    image_normalized = image.copy()
    for index in np.ndindex(image.shape[:-1]):
        spectrum = image[index]
        norm = np.linalg.norm(spectrum)
        if norm > 0:
            image_normalized[index] = spectrum / norm
    return image_normalized


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-m", "--mri", help="Input MRI image (ITK format)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-n", "--number", help="Number of components for dimension reduction", default=5)
parser.add_argument("-r", "--ratio", help="Compute ratio images (optional)", action="store_true")
parser.add_argument("-t", "--top", help="#Top (optional)", default=0)
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold (optional)", default=0)
parser.add_argument("--norm", help="Whether to normalize the spectra with respect to their L2 norm", action="store_true")
parser.add_argument("--norm_img", help="Normalization image filename (optional)")
parser.add_argument("--post_process", help="Post process with tSNE (optional)", action="store_true")
parser.add_argument("--split", help="For 3D volumes, process each 2D slice independently.", action="store_true")
parser.add_argument("--number_slice", help="Number of the slice to process", default=-1)
parser.add_argument("--memmap", help="Whether to store to memory map files.", action="store_true")
parser.add_argument("--no_translation", help="Whether to remove translation of NMF component images to match MR image", action="store_true")

args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
is_ratio = args.ratio
n = int(args.number)
top = int(args.top)
threshold = int(args.threshold)
normname = args.norm_img
is_norm = args.norm
post_process = args.post_process
is_split = args.split
number_slice = int(args.number_slice)
is_memmap = args.memmap
no_translation = args.no_translation
translate_components = not no_translation

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

# print("Mass-to-charge ratio=", mzs)


image = image[..., mzs >= threshold]

if is_norm:
    image = normalize(image)

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

if is_split and len(image.shape) == 4 and len(image_mri.shape) == 3:
    outroot, outext = os.path.splitext(outname)
    for k in range(image.shape[2]):
        if number_slice < 0 or (number_slice >= 0 and k == number_slice):
            current_image = image[:,:,k,:]
            current_image_mri = image_mri[..., k]
            current_name = outroot + "_" + str(k) + outext
            print(current_image.shape, current_image_mri.shape, current_name)
            statistical_analysis(current_name, current_image, current_image_mri, mzs, n, is_ratio, top, post_process, is_norm=True, is_denoise=True, is_memmap=is_memmap, translate_components=translate_components)
else:
    statistical_analysis(outname, image, image_mri, mzs, n, is_ratio, top, post_process, is_norm=True, is_denoise=True, is_memmap=is_memmap, translate_components=translate_components)

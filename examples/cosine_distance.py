"""
Statistical analysis
With various measures (UIQ ~~ SSIM, cosine)
And adaptable denoising conditions
"""

import numpy as np
import matplotlib.pyplot as plt
import esmraldi.imzmlio as imzmlio
import esmraldi.segmentation as seg
import esmraldi.fusion as fusion
import SimpleITK as sitk
import os

import scipy.spatial.distance as distance
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import argparse
import cv2 as cv

from esmraldi.sliceviewer import SliceViewer
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error
from skimage.filters import sobel
from scipy.ndimage import uniform_filter, median_filter, gaussian_filter
from skimage.filters import threshold_otsu
import skimage.restoration as skrestore

from sklearn.cluster import KMeans

from sewar import utils, rmse_sw


def uiq_single(GT,P,ws):
    N = ws**2
    window = np.ones((ws,ws))

    GT_sq = GT*GT
    P_sq = P*P
    GT_P = GT*P

    GT_sum = uniform_filter(GT, ws, mode="constant")
    P_sum =  uniform_filter(P, ws, mode="constant")
    GT_sq_sum = uniform_filter(GT_sq, ws, mode="constant")
    P_sq_sum = uniform_filter(P_sq, ws, mode="constant")
    GT_P_sum = uniform_filter(GT_P, ws, mode="constant")



    GT_P_sum_mul = GT_sum*P_sum
    GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
    numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
    denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
    denominator = denominator1*GT_P_sum_sq_sum_mul
    numerator[abs(numerator)<1e-10] = 0.0
    denominator[abs(denominator)<1e-10] = 0.0

    q_map = np.ones(denominator.shape)
    index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
    q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
    index = (denominator != 0)
    q_map[index] = numerator[index]/denominator[index]
    return q_map

def universal_image_quality_index(GT,P,ws=8):
    """calculates universal image quality index (uqi).

    :param GT: first (original) input image.
    :param P: second (deformed) input image.
    :param ws: sliding window size (default = 8).

    :returns:  float -- uqi value.
    """
    GT,P = utils._initial_check(GT,P)
    qmap = [uiq_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])]
    qmap = np.mean(qmap, axis=0)
    uiq_measure = np.mean(qmap)
    return uiq_measure, qmap

def multiscale_uiq(img1, img2, scales=[7, 15, 20, 30, 40]):
    uiq_measures = []
    for scale in scales:
        uiq_scale, _ = universal_image_quality_index(img1, img2, scale)
        uiq_measures.append(uiq_scale)
    return np.mean(uiq_measures)

def median_cosine(img1, img2, ws=7):
    median_im1 = uniform_filter(img1, (ws, ws, 0), mode="constant")
    # median_im2 = uniform_filter(img2, ws, mode="constant")
    flatten_m1 = fusion.flatten(median_im1, is_spectral=True)
    flatten_m2 = fusion.flatten(median_im2)
    print(flatten_m1.shape, flatten_m2.shape)
    cosine_sim = cosine_similarity(flatten_m1, flatten_m2)
    return cosine_sim

def extract_regions(image, k):
    old_shape = image.shape
    new_shape = (np.prod(image.shape), 1)
    image_linearized = np.reshape(image, new_shape)
    kmeans = KMeans(k, random_state=0).fit(image_linearized)
    labels = kmeans.labels_
    image_labels = labels.reshape(old_shape)
    return image_labels


def statistical_analysis(image, image_mri, mzs, outname, weights=None, is_cosine=True):
    image_flatten = fusion.flatten(image, is_spectral=True)
    image_mri_flatten = fusion.flatten(image_mri)

    # print(image.shape)
    # ind = np.argwhere(mzs == "833.33/701.3") or np.array([[0]])
    # print(ind)
    # maldi6_5 = image[..., ind]
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(maldi6_5[..., 0, 0])
    # ax[1].imshow(image_mri)
    # plt.show()
    # print(ssim(maldi6_5[..., 0, 0], image_mri))

    measures = []
    images_close = []
    if is_cosine:
        measures = cosine_similarity(image_flatten, image_mri_flatten)
    else:
        for i in range(image.shape[-1]):
            image_slice = image[..., i]
            wrmse, im = weighted_rmse(image_slice, image_mri, weights)
            measures.append([wrmse])
            # uiq, im = universal_image_quality_index(image_mri, image_slice, 8)
            # measures.append([uiq])
            images_close.append(im)

    indices = [[i for i in range(len(measures))] for measure in measures[0]]
    for i in range(len(indices)):
        indices[i].sort(key=lambda x:measures[x][i], reverse=False)

    indices_array = np.array(indices)

    closest_image = None
    if images_close:
        print(np.array(images_close).shape)
        closest_image = np.array(images_close)[indices[0][0]]
        print(closest_image.shape)

    similar_images_list, similar_mzs_list = [], []
    for i in range(len(indices)):
        similar_images = np.take(image, indices[i], axis=-1)
        similar_mzs = np.take(mzs, indices[i])
        ind = np.argwhere(similar_mzs == "833.33/701.3") or np.array([[-1]])
        closest_image_ax = np.array(images_close)[indices[0][ind.flatten()[0]]]
        similar_images_list.append(similar_images)
        similar_mzs_list.append(similar_mzs)
        print(ind)


    # indices_closest = fusion.closest_pixels_cosine(image_flatten[indices[0]]**2, image_mri_flatten[0]**2)
    # values = np.array([255-int((i*255)/len(indices_closest)) for i in range(len(indices_closest))])
    # maldi_closest = image_flatten[0]
    # maldi_closest[indices_closest] = values
    # maldi_closest = np.reshape(maldi_closest, image.shape[:-1])


    np.savetxt(outname, similar_mzs, delimiter=";", fmt="%s")
    print(np.array(similar_mzs_list)[:, :10])

    if len(similar_images.shape) == 3:
        image_display = closest_image is not None
        size = 5 + (1 if image_display else 0)
        print("size", size, image_display)
        fig, ax = plt.subplots(1, size)
        [axi.set_axis_off() for axi in ax.ravel()]
        ax[0].imshow(similar_images_list[0][..., 0], cmap="gray")
        ax[1].imshow(similar_images_list[0][..., 1], cmap="gray")
        ax[2].imshow(similar_images_list[0][..., 2], cmap="gray")

        if image_display:
            ax[-3].imshow(closest_image, cmap="gray")
        ax[-2].imshow(closest_image_ax, cmap="gray")
        ax[-1].imshow(image_mri, cmap="gray")
        name_img = os.path.splitext(outname)[0] + ".png"
        plt.savefig(name_img, dpi=300)
    elif len(similar_images.shape) == 4:
        fig, ax = plt.subplots(1, 4)
        tracker = SliceViewer(ax, np.transpose(similar_images[..., 0], (2, 1, 0)), np.transpose(similar_images_neighborhood[..., 0], (2, 1, 0)), np.transpose(image_mri, (2, 1, 0)), np.transpose(maldi_closest, (2, 1, 0)), vmin=0, vmax=255)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()

def weighted_rmse(image, image_mri, weights):
    diff_im = image - image_mri
    diff_im_sq = diff_im**2
    im = (diff_im * weights) ** 2
    return np.mean(im), np.sqrt(diff_im_sq)

def denoise_nlmeans(image, size, distance, spread):
    sigma_est = np.mean(skrestore.estimate_sigma(image, multichannel=True))
    patch_kw = dict(patch_size=size,      # s*s patches
                patch_distance=distance,  # d*d search area
                multichannel=True)
    image = image.copy(order='C')
    denoised = skrestore.denoise_nl_means(image, h=spread*sigma_est,fast_mode=False, **patch_kw)
    return denoised



"""
Computes the cosine distance between each MALDI ion image and the MRI image
And sorts the MALDI according to this distance in descending order
"""
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input MALDI image (imzML or nii)")
parser.add_argument("-m", "--mri", help="Input MRI image (ITK format)")
parser.add_argument("-o", "--output", help="Output image (ITK format)")
parser.add_argument("-g", "--threshold", help="Mass to charge ratio threshold", default=0)
parser.add_argument("-r", "--ratio", help="Compute ratio images", action="store_true")
parser.add_argument("--number_slice", help="Number of the slice to process (3D case)", default=-1)
parser.add_argument("-k", "--number_classes", help="Number of classes for k-means algorithm", default=0)
args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
threshold = int(args.threshold)
is_ratio = args.ratio
number_slice = int(args.number_slice)
k = int(args.number_classes)

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

image = image[..., mzs >= threshold]

mzs = mzs[mzs >= threshold]
mzs = np.around(mzs, decimals=2)
mzs = mzs.astype(str)

image_mri = sitk.GetArrayFromImage(sitk.ReadImage(mriname, sitk.sitkFloat32)).T
# if len(image.shape) == 3:
#     fig, ax = plt.subplots(1, 2)
#     ax[0].imshow(image[..., 0])
#     ax[1].imshow(image_mri)
#     plt.show()

# elif len(image.shape) == 4:
#     fig, ax = plt.subplots(1, 2)
#     display_maldi = np.transpose(image[..., 0], (2, 1, 0))
#     display_mri = np.transpose(image_mri, (2, 1, 0))
#     tracker = SliceViewer(ax, display_maldi, display_mri)
#     fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
#     plt.show()

if number_slice >= 0:
    image = image[..., number_slice, :]
    image_mri = image_mri[..., number_slice]


# image = median_filter(image, size=(5, 5, 1))
# image = gaussian_filter(image, sigma=(1,1,0))
# denoised = denoise_nlmeans(image, 5, 7, 10)

# fig, ax = plt.subplots(1, 2)
# display_maldi = np.transpose(image, (2, 1, 0))
# display_denoised = np.transpose(denoised, (2, 1, 0))
# tracker = SliceViewer(ax, display_maldi, display_denoised)
# fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
# plt.show()

image[image_mri == 0, :] = 0


if is_ratio:
    ratio_images, ratio_mzs = fusion.extract_ratio_images(image, mzs)
    image = np.concatenate((image, ratio_images), axis=-1)
    mzs = np.concatenate((mzs, ratio_mzs))
    image = ratio_images
    mzs = ratio_mzs
    indices_delete = fusion.remove_indices(image)
    image = np.delete(image, indices_delete, axis=-1)
    mzs = np.delete(mzs, indices_delete)
    print(image.shape)



image = imzmlio.normalize(image)
# image = np.uint8(cv.normalize(image, None, 0, 255, cv.NORM_MINMAX))
image_mri = imzmlio.normalize(image_mri)
image = image.astype(np.float64)
image_mri = image_mri.astype(np.float64)

for index in range(image.shape[-1]):
    current_image  = image[..., index]
    norm =  np.linalg.norm(current_image)
    if norm > 0:
        current_image /= norm
        image[..., index] = current_image

mask = np.ones_like(image_mri)
mask[image_mri == 0] = 0
weights = sobel(image_mri, mask=mask)
threshold_max = 100
weights[weights > threshold_max] = threshold_max
weights[weights < 1] = 1
# threshold = threshold_otsu(weights)
# weights = np.where(weights > threshold, 5, 1)
# plt.imshow(weights)
# plt.show()

image_mri /= np.linalg.norm(image_mri)


if k > 0:
    image_labels = extract_regions(image_mri, k)
    fig, ax = plt.subplots()
    ax.imshow(image_labels)
    plt.show()
    print("Please enter the label of the grain")
    grain_label = int(input())
    for i in range(k):
        image_labels_repeated = np.repeat(image_labels[..., np.newaxis], image.shape[-1], axis=-1)
        condition = (image_labels_repeated == i) | (image_labels_repeated == grain_label)
        condition_mri = (image_labels == i) | (image_labels == grain_label)
        current_image = np.where(condition, image.copy(), 0)
        current_image_mri = np.where(condition_mri, image_mri.copy(), 0)
        root, ext = os.path.splitext(outname)
        current_outname = root + "_" + str(i) + ext
        statistical_analysis(current_image, current_image_mri, mzs, current_outname, weights=weights, is_cosine=False)

else:
    statistical_analysis(image, image_mri, mzs, outname, weights=1, is_cosine=False)

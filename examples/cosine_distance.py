import numpy as np
import matplotlib.pyplot as plt
import esmraldi.imzmlio as imzmlio
import esmraldi.segmentation as seg
import esmraldi.fusion as fusion
import SimpleITK as sitk
import scipy.spatial.distance as distance
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import argparse
import cv2 as cv

from esmraldi.sliceviewer import SliceViewer
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error

from sewar import uqi, ssim

def uiq_measure(img1, img2):
    mean_im1 = np.mean(img1)
    mean_im2 = np.mean(img2)
    var_im1 = np.var(img1)
    var_im2 = np.var(img2)
    cov = np.sum((img1 - mean_im1)*(img2-mean_im2))/(len(img1) - 1)
    num = 4 * cov * mean_im1 * mean_im2
    den = (var_im1 ** 2 + var_im2 ** 2) * (mean_im1**2 + mean_im2**2)
    if den != 0:
        return num/den

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

args = parser.parse_args()

inputname = args.input
mriname = args.mri
outname = args.output
threshold = int(args.threshold)
is_ratio = args.ratio
number_slice = int(args.number_slice)


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

# image = imzmlio.normalize(image)
image = np.uint8(cv.normalize(image, None, 0, 255, cv.NORM_MINMAX))
image_mri = imzmlio.normalize(image_mri)

if is_ratio:
    ratio_images, ratio_mzs = fusion.extract_ratio_images(image, mzs)
    image = np.concatenate((image, ratio_images), axis=-1)
    mzs = np.concatenate((mzs, ratio_mzs))
    image = ratio_images
    mzs = ratio_mzs
    print(image.shape)

image_flatten = fusion.flatten(image, is_spectral=True)
image_mri_flatten = fusion.flatten(image_mri)

print(image.shape)
ind = np.argwhere(mzs == "833.33/701.3")
print(ind)
# maldi6_5 = image[..., ind]
# fig, ax = plt.subplots(1,2)
# ax[0].imshow(maldi6_5[..., 0, 0])
# ax[1].imshow(image_mri)
# plt.show()
# print(ssim(maldi6_5[..., 0, 0], image_mri))

# cosines = cosine_similarity(image_flatten, image_mri_flatten)
cosines = []

for i in range(image.shape[-1]):
    image_slice = image[..., i]
    uiq = ssim(image_mri, image_slice, 20)[0]
    cosines.append([uiq])

indices = [[i for i in range(len(cosines))] for measure in cosines[0]]
print(len(indices))
for i in range(len(indices)):
    indices[i].sort(key=lambda x:cosines[x][i], reverse=True)

indices_array = np.array(indices)
print(indices_array.shape)

similar_images_list, similar_mzs_list = [], []
for i in range(len(indices)):
    similar_images = np.take(image, indices[i], axis=-1)
    similar_mzs = np.take(mzs, indices[i])
    similar_images_list.append(similar_images)
    similar_mzs_list.append(similar_mzs)
    print(indices[i][ind.flatten()[0]])


# cosines_neighborhood = fusion.cosine_neighborhood(image, image_mri, 1)
# indices_neighborhood = [i for i in range(len(cosines_neighborhood))]
# indices_neighborhood.sort(key=lambda x:cosines_neighborhood[x][0], reverse=True)
# indices_array = np.array(indices_neighborhood)
# similar_images_neighborhood = np.take(image, indices_neighborhood, axis=-1)
# similar_mzs_neighborhood = np.take(mzs, indices_neighborhood)

# indices_closest = fusion.closest_pixels_cosine(image_flatten[indices[0]]**2, image_mri_flatten[0]**2)
# values = np.array([255-int((i*255)/len(indices_closest)) for i in range(len(indices_closest))])
# maldi_closest = image_flatten[0]
# maldi_closest[indices_closest] = values
# maldi_closest = np.reshape(maldi_closest, image.shape[:-1])


np.savetxt(outname, similar_mzs, delimiter=";", fmt="%s")
print(np.array(similar_mzs_list)[:, :10])
# print(similar_mzs_neighborhood)

if len(similar_images.shape) == 3:
    fig, ax = plt.subplots(1, len(indices) + 1)
    for i  in range(len(similar_images_list)):
        ax[i].imshow(similar_images_list[i][..., 0], cmap="gray")
    # ax[1].imshow(similar_images_neighborhood[..., 0], cmap="gray")
    ax[-1].imshow(image_mri, cmap="gray")
    # ax[3].imshow(maldi_closest, cmap="gray")
    plt.show()
elif len(similar_images.shape) == 4:
    fig, ax = plt.subplots(1, 4)
    tracker = SliceViewer(ax, np.transpose(similar_images[..., 0], (2, 1, 0)), np.transpose(similar_images_neighborhood[..., 0], (2, 1, 0)), np.transpose(image_mri, (2, 1, 0)), np.transpose(maldi_closest, (2, 1, 0)), vmin=0, vmax=255)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

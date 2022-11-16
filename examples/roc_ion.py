import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import sklearn.metrics as metrics
import esmraldi.imageutils as imageutils
import SimpleITK as sitk
import matplotlib.pyplot as plt
import xlsxwriter
import os

from skimage.color import rgb2gray



def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("--mz", help="M/Z", nargs="+", type=float)
parser.add_argument("-w", "--weight", help="Weight ROC by amount of points in each condition", action="store_true")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
region_names = args.regions
normalization = float(args.normalization)
mz = args.mz
is_weighted = args.weight

unique_image = False
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
    pathcsv = os.path.splitext(input_name)[0] + ".csv"
    if os.path.isfile(pathcsv):
        mzs = np.loadtxt(pathcsv)
    else:
        mzs = np.array([0])
        unique_image = True


mask = read_image(mask_name)
regions = []
for region_name in region_names:
    region = read_image(region_name).astype(bool)
    regions.append(region)

n = len(np.where(mask>0)[0])



name = "No norm"
if normalization > 0:
    name = str(normalization)



norm_img = None
if normalization > 0:
    norm_img = imageutils.get_norm_image(images, normalization, mzs)

shape = images.shape[:-1]
if unique_image:
    shape = images.shape

indices, indices_ravel = fusion.roc_indices(mask, shape, norm_img)

region_bool = fusion.region_to_bool(regions, indices_ravel, shape)

if unique_image:
    current_image = images
else:
    region_bool = fusion.region_to_bool(regions, indices_ravel, shape)

    closest_mz_indices = []
    for m in mz:
        closest_mz_index = np.abs(mzs - m).argmin()
        closest_mz_indices.append(closest_mz_index)
        print("Found", mzs[closest_mz_index], mz)
    closest_mz_indices = np.array(closest_mz_indices)

    current_image = images[..., closest_mz_indices]

colors = ["b", "r", "g"]
for i in range(current_image.shape[-1]):
    c = current_image[..., i]
    if normalization > 0:
        c = imageutils.normalize_image(c, norm_img)


    sub_region = c[indices]
    current_values = sub_region.flatten()

    for j, binary_label in enumerate(region_bool):
        fpr, tpr, thresholds = fusion.roc_curve(binary_label, current_values, is_weighted=is_weighted)
        nb_ones = np.count_nonzero(binary_label)
        nb_zeros = np.count_nonzero(~binary_label)
        current_values = current_values.astype(np.float64)
        # A = current_values.max() - current_values
        # ppv, recall, _ = precision_recall_curve(binary_label, current_values)
        # npv, recall2, _ = precision_recall_curve(binary_label, A, pos_label=0)
        plt.plot(fpr, tpr, color=colors[j], label=region_names[j])
        plt.plot([0,1], [0,1], "--", c="k")
        plt.xlim((0, 1))
        plt.ylim((0, 1))
        plt.title(mzs[closest_mz_indices[i]])
        print("Cutoff", fusion.single_roc_cutoff(c, indices, [binary_label], lambda fpr, tpr, thresholds: fusion.cutoff_generalized_youden(fpr, tpr, thresholds, nb_zeros, nb_ones), is_weighted=is_weighted))
        print(fusion.single_roc_auc(c, indices, [binary_label], is_weighted=is_weighted))
    plt.show()

# plt.plot([0, 1], [0, 1], color="k", linestyle="--")
plt.show()

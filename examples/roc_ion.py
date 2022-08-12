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
    mask = rgb2gray(mask)
    mask = mask.T
    return mask


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("--mz", help="M/Z")
parser.add_argument("-w", "--weight", help="Weight ROC by amount of points in each condition", action="store_true")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
region_names = args.regions
normalization = float(args.normalization)
mz = float(args.mz)
is_weighted = args.weight

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
    mzs = np.loadtxt(os.path.splitext(input_name)[0] + ".csv")



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

indices, indices_ravel = fusion.roc_indices(mask, images.shape[:-1], norm_img)


region_bool = fusion.region_to_bool(regions, indices_ravel, images.shape[:-1])

closest_mz_index = np.abs(mzs - mz).argmin()
print("Found", mzs[closest_mz_index], mz)

current_image = images[..., closest_mz_index]
if normalization > 0:
    current_image = imageutils.normalize_image(current_image, norm_img)


sub_region = current_image[indices]
current_values = sub_region.flatten()

colors = ["g", "r", "b"]

for j, binary_label in enumerate(region_bool):
    fpr, tpr, thresholds = fusion.roc_curve(binary_label, current_values, is_weighted=is_weighted)
    current_values = current_values.astype(np.float64)
    # A = current_values.max() - current_values
    # ppv, recall, _ = precision_recall_curve(binary_label, current_values)
    # npv, recall2, _ = precision_recall_curve(binary_label, A, pos_label=0)
    plt.plot(fpr, tpr, color=colors[j], label=region_names[j])
    print(fusion.single_roc_auc(current_image, indices, [binary_label], is_weighted=is_weighted))

# plt.plot([0, 1], [0, 1], color="k", linestyle="--")
plt.show()

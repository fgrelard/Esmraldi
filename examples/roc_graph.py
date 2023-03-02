import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
import esmraldi.imageutils as imageutils
import SimpleITK as sitk
import matplotlib.pyplot as plt
import xlsxwriter
import os
import mplcursors

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
parser.add_argument("-o", "--output", help="Output .xlsx files with stats")
parser.add_argument("-w", "--weight", help="Weight ROC by amount of points in each condition", action="store_true")
parser.add_argument("-f", "--function", help="Cutoff function", default="distance")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
region_names = args.regions
output_name = args.output
normalization = float(args.normalization)
is_weighted = args.weight
function = args.function

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

if function == "distance":
    print("Choosing distance cutoff function")
    fn = fusion.cutoff_distance
else:
    print("Choosing half TPR cutoff function")
    fn = fusion.cutoff_half_tpr

print(images.shape)


name = "No norm"
if normalization > 0:
    name = str(normalization)


mask = read_image(mask_name)
regions = []
for region_name in region_names:
    region = read_image(region_name)
    regions.append(region)

n = len(np.where(mask>0)[0])

norm_img = None
if normalization > 0:
    norm_img = imageutils.get_norm_image(images, normalization, mzs)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[...,i], norm_img)

averages = np.mean(images, axis=(0,1))

indices, indices_ravel = fusion.roc_indices(mask, images.shape[:-1], norm_img)


region_bool = fusion.region_to_bool(regions, indices_ravel, images.shape[:-1])
roc_auc_scores = fusion.roc_auc_analysis(images, indices, region_bool, norm_img, is_weighted=is_weighted)
roc_cutoffs_youden = fusion.roc_cutoff_analysis(images, indices, region_bool, is_weighted=is_weighted, fn=fusion.cutoff_generalized_youden)


plt.scatter(roc_auc_scores, roc_cutoffs_youden)
plt.xlabel("ROC-AUC")
plt.ylabel(function)
mplcursors.cursor(multiple=True).connect("add", lambda sel: sel.annotation.set_text("{:.3f}".format(mzs[sel.index])))
plt.show()

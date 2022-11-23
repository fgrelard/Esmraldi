import argparse
import numpy as np
import esmraldi.imzmlio as io
import esmraldi.fusion as fusion
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
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)", default=None)
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output .xlsx files with stats")
parser.add_argument("--cutoffs", action="store_true", help="Compute cutoff analysis")
parser.add_argument("-w", "--weight", help="Weight ROC by amount of points in each condition", action="store_true")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
region_names = args.regions
output_name = args.output
normalization = float(args.normalization)
is_weighted = args.weight
is_cutoffs = args.cutoffs

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


workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})

name = "No norm"
if normalization > 0:
    name = str(normalization)

worksheet = workbook.add_worksheet(name)

worksheets = [worksheet,
              workbook.add_worksheet("Averages"),
              workbook.add_worksheet("Averages per")]

if is_cutoffs:
    worksheets += [workbook.add_worksheet("Distance"),
                   workbook.add_worksheet("Generalized Youden"),
                   workbook.add_worksheet("Half TPR"),
                   workbook.add_worksheet("Efficiency")]


if mask_name is not None:
    mask = read_image(mask_name)
else:
    mask = np.ones_like(images[..., 0])

regions = []
print("Read image")
for region_name in region_names:
    region = read_image(region_name)
    regions.append(region)

n = len(np.where(mask>0)[0])

norm_img = None

print("normalization", normalization)
if normalization > 0:
    norm_img = imageutils.get_norm_image(images, normalization, mzs)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[...,i], norm_img)


indices, indices_ravel = fusion.roc_indices(mask, images.shape[:-1], norm_img)


for worksheet in worksheets:
    for i, region in enumerate(regions):
        region_name = region_names[i]
        name = os.path.splitext(os.path.basename(region_name))[0]
        worksheet.write(i+1, 0, name, header_format)
    worksheet.freeze_panes(1, 1)

print("Starting ROC AUC")
region_bool = fusion.region_to_bool(regions, indices_ravel, images.shape[:-1])

averages = np.mean(images, axis=(0,1))[:, np.newaxis]
print(averages.shape)

averages_per = fusion.averages_per_region(images, indices, region_bool)

print("av per", averages_per.shape)

roc_auc_scores = fusion.roc_auc_analysis(images, indices, region_bool, norm_img, is_weighted=is_weighted)

L = [roc_auc_scores, averages, averages_per]

print("End AUC")

if is_cutoffs:
    print("Cutoff distance")
    distances = fusion.roc_cutoff_analysis(images, indices, region_bool, is_weighted=is_weighted, fn=fusion.cutoff_distance2)
    print("Cutoff Youden")
    youden = fusion.roc_cutoff_analysis(images, indices, region_bool, is_weighted=is_weighted, fn=fusion.cutoff_generalized_youden)
    print("Cutoff Half-TPR")
    half_tpr = fusion.roc_cutoff_analysis(images, indices, region_bool, is_weighted=is_weighted, fn=fusion.cutoff_half_tpr)
    print("Cutoff Efficiency")
    efficiency = fusion.roc_cutoff_analysis(images, indices, region_bool, is_weighted=is_weighted, fn=fusion.cutoff_efficiency)
    L += [distances, youden, half_tpr, efficiency]

for worksheet_index, values in enumerate(L):
    for (i, j), individual_value in np.ndenumerate(values):
        worksheets[worksheet_index].write(0, i+1, mzs[i], header_format)
        worksheets[worksheet_index].write(j+1, i+1, individual_value)

# for (i, j), auc in np.ndenumerate(roc_auc_scores):
#     worksheet.write(0, i+1, mzs[i], header_format)
#     worksheet.write(j+1, i+1, auc)

workbook.close()

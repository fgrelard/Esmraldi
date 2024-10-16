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
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-n", "--normalization", help="Normalization w.r.t. to given m/z", default=0)
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
region_names = args.regions
output_name = args.output
normalization = float(args.normalization)

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
    if region_name == mask_name:
        continue
    region = read_image(region_name)
    regions.append(region)

n = len(np.where(mask>0)[0])

print(n)

workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
workbook.use_zip64()
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
worksheet_stats = workbook.add_worksheet("Stats")

worksheets = []
worksheets.append((worksheet, worksheet_stats))


norm_img = None
if normalization > 0:
    norm_img = imageutils.get_norm_image(images, normalization, mzs)
    for i in range(images.shape[-1]):
        images[..., i] = imageutils.normalize_image(images[...,i], norm_img)

indices, indices_ravel = fusion.roc_indices(mask, images.shape[:-1], norm_img)


region_bool = fusion.region_to_bool(regions, indices_ravel, images.shape[:-1])
print(len(region_bool))
sub = False
if sub:
    mzs_target = [837.549, 863.56,
                  773.534, 771.51,
                  885.549, 437.2670,
                  871.57, 405.2758,#dispersion
                  859.531372070312, 714.5078, #LB
                  644.5015869, 715.5759, #LT
                  287.0937, 296.0824, 746.512]
    indices_mz = [np.abs(mzs - mz).argmin() for mz in mzs_target]
    images = images[..., indices_mz]
    mzs = mzs[indices_mz]

for w, (worksheet, worksheet_stats) in enumerate(worksheets):
    worksheet.write(0, 0, "Pixel number", header_format)
    worksheet.write_column(1, 0, indices_ravel)
    for i, region in enumerate(regions):
        region_name = region_names[i]
        name = os.path.splitext(os.path.basename(region_name))[0]
        worksheet.write(0, i+1, name, header_format)
        worksheet.write_column(1, i+1, region_bool[i])

    worksheet_stats.write_column(0, 0, ["m/z", "Mean", "Stddev", "N"])
    worksheet.freeze_panes(1, 1)
    worksheet_stats.freeze_panes(1, 1)

nreg = len(regions)
for i in range(images.shape[-1]):
    mz = mzs[i]
    current_image = images[..., i]


    sub_region = current_image[indices]

    current_values = sub_region.flatten()

    mean = np.mean(sub_region)
    stddev = np.std(sub_region)

    worksheet.write(0, i+nreg+1, mz, header_format)
    worksheet.write_column(1, i+nreg+1, current_values)

    worksheet_stats.write(0, i+1, mz, header_format)
    worksheet_stats.write_column(1, i+1, [mean, stddev, n])
workbook.close()

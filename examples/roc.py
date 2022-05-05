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
from sklearn.metrics import roc_auc_score


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
wparser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
region_names = args.regions
output_name = args.output
normalization = float(args.normalization)

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

worksheets = []
worksheets.append(worksheet)

mask = read_image(mask_name)
regions = []
for region_name in region_names:
    region = read_image(region_name)
    regions.append(region)

n = len(np.where(mask>0)[0])

norm_img = None
if normalization > 0:
    norm_img = imageutils.get_norm_image(images, normalization, mzs)

indices, indices_ravel = fusion.roc_indices(mask, (max_x, max_y), norm_img)

for worksheet in worksheets:
    for i, region in enumerate(regions):
        region_name = region_names[i]
        name = os.path.splitext(os.path.basename(region_name))[0]
        worksheet.write(i+1, 0, name, header_format)
    worksheet.freeze_panes(1, 1)

region_bool = fusion.region_to_bool(regions, indices_ravel, (max_x, max_y))
roc_auc_scores = fusion.roc_auc_analysis(images, indices, region_bool, norm_img)

for (i, j), auc in np.ndenumerate(roc_auc_scores):
    worksheet.write(0, i+1, mzs[i], header_format)
    worksheet.write(j+1, i+1, auc)

workbook.close()

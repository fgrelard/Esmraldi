import argparse
import numpy as np
import esmraldi.imzmlio as io
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

def find_indices(image, shape):
    indices = np.where(image > 0)
    return np.ravel_multi_index(indices, shape, order='F')

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
region_names = args.regions
output_name = args.output

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

mask = read_image(mask_name)
regions = []
for region_name in region_names:
    region = read_image(region_name)
    regions.append(region)

n = len(np.where(mask>0)[0])

print(n)
indices = np.where(mask > 0)
indices_ravel = find_indices(mask, (max_x, max_y))
workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})

worksheet = workbook.add_worksheet("No norm")

worksheets = []
worksheets.append(worksheet)


region_bool = []
for region in regions:
    indices_regions = find_indices(region, (max_x, max_y))
    inside_region = np.in1d(indices_ravel, indices_regions).astype(int)
    is_same = np.all(inside_region == inside_region[0])
    if not is_same:
        region_bool.append(inside_region)

for worksheet in worksheets:
    for i, region in enumerate(regions):
        region_name = region_names[i]
        name = os.path.splitext(os.path.basename(region_name))[0]
        worksheet.write(i+1, 0, name, header_format)
    worksheet.freeze_panes(1, 1)


nreg = len(regions)
for i in range(images.shape[-1]):
    mz = mzs[i]
    current_image = images[..., i]
    sub_region = current_image[indices]
    current_values = sub_region.flatten()
    worksheet.write(0, i+1, mz, header_format)

    for j, binary_label in enumerate(region_bool):
        auc = roc_auc_score(binary_label, current_values)
        worksheet.write(j+1, i+1, auc)
workbook.close()

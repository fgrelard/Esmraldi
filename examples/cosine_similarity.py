import argparse
import numpy as np
import esmraldi.imzmlio as io
import SimpleITK as sitk
import matplotlib.pyplot as plt
import xlsxwriter
import os

from skimage.color import rgb2gray
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
import csv
import esmraldi.fusion as fusion
import scipy.stats as stats
from scipy.spatial.distance import cosine

def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask




parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input ITK")
parser.add_argument("--mzs", help="m/z values associated to input")
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-r", "--regions", help="Subregions inside mask", nargs="+", type=str)
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
mzs_name = args.mzs
mask_name = args.mask
region_names = args.regions
output_name = args.output

images = sitk.GetArrayFromImage(sitk.ReadImage(input_name)).T
with open(mzs_name, "r", encoding="utf-8-sig") as f:
    reader = list(csv.reader(f, delimiter=","))
    mzs = np.array([row[0] for row in reader])

print(mzs, len(mzs))

num_value = "833.33"
denom_value = "701.3"
num = mzs==num_value
denom = mzs==denom_value

if num.any() and denom.any():
    image_num = images[..., num].astype(np.float64)
    image_denom = images[..., denom].astype(np.float64)

    divided = np.zeros_like(image_num)
    np.divide(image_num, image_denom, out=divided, where=image_denom!=0)
    images = np.dstack((divided, images))
    mzs = np.concatenate(([num_value + "/" + denom_value], mzs))
    print(images.shape)


mask = read_image(mask_name)
regions = []
for region_name in region_names:
    region = read_image(region_name)
    regions.append(region)

n = len(np.where(mask>0)[0])

indices = np.where(mask > 0)
nreg = len(regions)
mask_flatten = fusion.flatten(mask)
images = io.normalize(images)
image_flatten = fusion.flatten(images, is_spectral=True)


workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
workbook.use_zip64()
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})

name = "Original MS"
second_name = "Masked MS"

worksheet_original = workbook.add_worksheet(name)
worksheet_mask = workbook.add_worksheet(second_name)
# region_bool = fusion.region_to_bool(regions, indices_ravel, (max_x, max_y))

for i, worksheet in enumerate(workbook.worksheets()):
    worksheet.write_row(0, 1, mzs, header_format)
    worksheet.write(1, 0, "Full image", header_format)
    for j, region in enumerate(regions):
        region_name = region_names[j]
        name = os.path.splitext(os.path.basename(region_name))[0]
        worksheet.write(j+2, 0, name, header_format)

    worksheet.freeze_panes(1, 1)
    sim = cosine_similarity(image_flatten, mask_flatten)
    worksheet.write_row(1, 1, sim)

    for j, r in enumerate(regions):
        mask_region = mask.copy()
        cond = (r == 0) & (regions[0] == 0)
        mask_region[cond] = 0
        mask_region_flatten = fusion.flatten(mask_region)
        images_region = images.copy()
        if i == 1:
            images_region[cond, ...] = 0
        region_flatten = fusion.flatten(r)
        image_region_flatten = fusion.flatten(images_region, is_spectral=True)
        sim = cosine_similarity(image_region_flatten, mask_region_flatten)
        worksheet.write_row(j+2, 1, sim)

workbook.close()

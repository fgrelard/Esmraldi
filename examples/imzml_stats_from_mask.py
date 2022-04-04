import argparse
import numpy as np
import esmraldi.imzmlio as io
import SimpleITK as sitk
import matplotlib.pyplot as plt
import xlsxwriter
import os

from skimage.color import rgb2gray

def get_norm_image(images, norm, mzs):
    if norm == "tic":
        img_norm = np.sum(images, axis=-1)
    else:
        closest_mz_index = np.abs(mzs - norm).argmin()
        img_norm = images[..., closest_mz_index]

    return img_norm

def process_image(current_img, img_norm):
    return_img = np.zeros_like(current_img)
    np.divide(current_img, img_norm, out=return_img, where=img_norm!=0)
    return return_img

def read_image(image_name):
    sitk.ProcessObject_SetGlobalWarningDisplay(False)
    mask = sitk.GetArrayFromImage(sitk.ReadImage(image_name))
    if mask.ndim > 2:
        mask = rgb2gray(mask)
    mask = mask.T
    return mask

def find_indices(image, shape):
    indices = np.where(image > 0)
    return np.ravel_multi_index(indices, shape, order='F')

def region_labels(regions, indices, shape):
    region_bool = []
    for region in regions:
        indices_regions = find_indices(region, (max_x, max_y))
        inside_region = np.in1d(indices, indices_regions)
        region_bool.append(inside_region)
    return region_bool

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
workbook.use_zip64()
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})

worksheet = workbook.add_worksheet("No norm")
worksheet_stats = workbook.add_worksheet("Stats")

worksheets = []
worksheets.append((worksheet, worksheet_stats))

normalization = [828.7287, 812.7540, 790.773, "tic"]
normalization = [703.669]

indices_2D = [indices]

region_label = region_labels(regions, indices_ravel, (max_x, max_y))
region_bool = [region_label]

normalization_images = []
for norm in normalization:
    worksheet = workbook.add_worksheet(str(norm))
    worksheet_stats = workbook.add_worksheet("Stats " + str(norm))
    worksheets.append((worksheet, worksheet_stats))

    norm_img = get_norm_image(images, norm, mzs)
    norm_indices = find_indices(norm_img, (max_x, max_y))
    norm_indices = np.intersect1d(indices_ravel, norm_indices)
    norm_indices_2D = np.unravel_index(norm_indices, (max_x, max_y), order="F")
    region_label = region_labels(regions, norm_indices, (max_x, max_y))
    normalization_images.append(norm_img)
    indices_2D.append(norm_indices_2D)
    region_bool.append(region_label)


for w, (worksheet, worksheet_stats) in enumerate(worksheets):
    worksheet.write(0, 0, "Pixel number", header_format)
    rindex = np.ravel_multi_index(indices_2D[w], (max_x, max_y), order="F")
    worksheet.write_column(1, 0, rindex)
    for i, region in enumerate(regions):
        region_name = region_names[i]
        name = os.path.splitext(os.path.basename(region_name))[0]
        worksheet.write(0, i+1, name, header_format)
        worksheet.write_column(1, i+1, region_bool[w][i])

    worksheet_stats.write_column(0, 0, ["m/z", "Mean", "Stddev", "N"])
    worksheet.freeze_panes(1, 1)
    worksheet_stats.freeze_panes(1, 1)

closest_mz_indices = [np.abs(mzs - norm).argmin() for norm in normalization[:-1:]]
print(closest_mz_indices)

nreg = len(regions)
for i in range(images.shape[-1]):
    mz = mzs[i]
    current_image = images[..., i]

    for j, (worksheet, worksheet_stats) in enumerate(worksheets):
        if j > 0:
            current_image = process_image(current_image, normalization_images[j-1])

        current_indices = indices_2D[j]
        sub_region = current_image[current_indices]

        current_values = sub_region.flatten()

        mean = np.mean(sub_region)
        stddev = np.std(sub_region)

        worksheet.write(0, i+nreg+1, mz, header_format)
        worksheet.write_column(1, i+nreg+1, current_values)

        worksheet_stats.write(0, i+1, mz, header_format)
        worksheet_stats.write_column(1, i+1, [mean, stddev, n])
    if i > 10:
        break
workbook.close()

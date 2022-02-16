import argparse
import numpy as np
import esmraldi.imzmlio as io
import SimpleITK as sitk
import matplotlib.pyplot as plt
import xlsxwriter

from skimage.color import rgb2gray

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", help="Input .imzML")
parser.add_argument("-m", "--mask", help="Mask image (any ITK format)")
parser.add_argument("-o", "--output", help="Output .csv files with stats")
args = parser.parse_args()

input_name = args.input
mask_name = args.mask
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
mask = sitk.GetArrayFromImage(sitk.ReadImage(mask_name))
mask = rgb2gray(mask)
mask = mask.T

n = len(np.where(mask>0)[0])

print(n)
indices = np.where(mask > 0)
workbook = xlsxwriter.Workbook(output_name, {'strings_to_urls': False})
header_format = workbook.add_format({'bold': True,
                                     'align': 'center',
                                     'valign': 'vcenter',
                                     'fg_color': '#D7E4BC',
                                     'border': 1})

left_format = workbook.add_format({'align': 'left'})

worksheet = workbook.add_worksheet("Full data")
worksheet_stats = workbook.add_worksheet("Stats")
worksheet_stats.write_column(0, 0, ["m/z", "Mean", "Stddev", "N"])

worksheet.freeze_panes(1, 0)
worksheet_stats.freeze_panes(1, 1)

for i in range(images.shape[-1]):
    mz = mzs[i]
    current_image = images[..., i]
    sub_region = current_image[indices]

    mean_region = np.mean(sub_region)
    stddev_region = np.std(sub_region)
    values = sub_region.flatten()

    mean = np.mean(sub_region)
    stddev = np.std(sub_region)

    worksheet.write(0, i, mz, header_format)
    worksheet.write_column(1, i, values)

    worksheet_stats.write(0, i+1, mz, header_format)
    worksheet_stats.write_column(1, i+1, [mean, stddev, n])

workbook.close()
